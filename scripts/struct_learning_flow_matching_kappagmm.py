# we implement something beyond 2d here.
# we save the infered smaples at intermediate points in time here.

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import tyro
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import Module
from tqdm.auto import tqdm

from flow_matching.visualization import save_projections_as_gif, plot_loss_curve
from flow_matching.datasets import TOY_DATASETS, SyntheticDataset, ToyDatasetName, DatasetkappaGMM
from flow_matching.solver import TimeBroadcastWrapper,run_ode, sample_ode
from flow_matching.utils import set_seed
import numpy as np


@dataclass
class ScriptArguments:
    dim: int = 2
    kappa:float = 0.3
    output_dir: Path = Path("outputs")
    learning_rate: float = 1e-3
    batch_size: int = 4096
    iterations: int = 20000
    nb_log_points: int = 10
    log_scale: str = 'linear'
    log_every: int = 2000
    hidden_dim: int = 512
    seed: int = 42
    interval:int =1


class Mlp(Module):
    def __init__(self, dim: int = 2, time_dim: int = 1, h: int = 64) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim + time_dim, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, dim),
        )

    def forward(
        self,
        x_t: Float[Tensor, "batch dim"],
        t: Float[Tensor, "batch time_dim"],
    ) -> Float[Tensor, "batch dim"]:
        h = torch.cat([x_t, t], dim=1)
        return self.layers(h)


def main(args: ScriptArguments) -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda") 
    else:
        try : 
            device = torch.device("mps")
        except :
            device = torch.device("cpu")

    set_seed(args.seed)
    output_dir = args.output_dir / "cfm" / f"kappagmm_{args.dim}_{args.kappa}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"GMM Dataset: {args.dim}, {args.kappa}")

    dataset = DatasetkappaGMM(dim=args.dim,device=device, kappa=args.kappa)

    # main directions
    m1 = (dataset.mus[0] + dataset.mus[1]) / 2
    m2 = (dataset.mus[0] - dataset.mus[1]) / 2


    flow = Mlp(dim=dataset.dim, time_dim=1, h=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(flow.parameters(), args.learning_rate)

    # Training        
    if args.log_scale == 'log':
        log_steps = set(np.unique(np.logspace(0, np.log10(args.iterations), num=args.nb_log_points).astype(int)) - 1)
    else :
        log_steps = set(np.unique(np.linspace(0,args.iterations,num=args.nb_log_points, dtype=int)))
        

    projections = []
    losses = []
    eval_times = []
    for global_step in tqdm(range(args.iterations), desc="Training", dynamic_ncols=True):
        x_1 = dataset.sample(args.batch_size)
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.size(0), 1, device=device)

        # Compute the Conditional Flow Matching objective
        # Check eq. (22) and (23) in the paper: https://arxiv.org/abs/2210.02747
        # where, we set \sigma_{\min} = 0
        x_t = (1 - t) * x_0 + t * x_1  # \phi_t(x_0)
        dx_t = x_1 - x_0  # u_t(x|x_t)

        optimizer.zero_grad()
        loss = F.mse_loss(flow(x_t=x_t, t=t), dx_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if global_step in log_steps :
            tqdm.write(f"| step: {global_step + 1:6d} | loss: {loss.item():8.4f} |")

            # do an ode run and add the projection on mus
            flow.eval()

            wrapped_model = TimeBroadcastWrapper(flow)
            
            sampled_data_ = sample_ode(
            flow=wrapped_model,
            dim=args.dim,
            num_samples=int(1e4) # hard code, sufficient for our needs
            # filename=f"ode_sampling_evolution_{args.dataset}.png",
            )

            p1 = sampled_data_ @ m1 / args.dim # (nsamples)
            p2 = sampled_data_ @ m2 / args.dim # (nsamples)
            
            projections_ = torch.vstack((p1, p2)).T.detach() # (nsamples, 2)
            projections.append(projections_.cpu())

            eval_times.append(global_step)

            flow.train()

    flow.eval()
    torch.save(flow.state_dict(), output_dir / "ckpt.pth")
    plot_loss_curve(losses=losses, output_path=output_dir / "losses.png")
    
    # save the mus
    torch.save(dataset.mus, output_dir / "mus.pt")

    # save the intermediate generated smaples
    projections = torch.stack(projections) #(nb_eval_steps, nbsamples, 2)
    eval_times = torch.tensor(eval_times) #(nb_eval_steps)

    # save these intermediate projections as .pt dir
    
    projection_data= {'projections' : projections, 'eval_times' : eval_times}
    
    torch.save(
        projection_data,
        output_dir / 'basis_proj.pt'
    )
    print(f'saved projection data at {output_dir / 'basis_proj.pt'}')

    # make a nice gif with these projections
    save_projections_as_gif(
        projection_data,
        dataset.mus,
        args.dim,
        args.kappa,
        output_dir=output_dir,
        interval=args.interval
        )

if __name__ == "__main__":
    main(tyro.cli(ScriptArguments))
