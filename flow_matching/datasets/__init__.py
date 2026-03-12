from typing import Literal

from flow_matching.datasets.synthetic_datasets import (
    DatasetCheckerboard,
    DatasetInvertocat,
    DatasetMixture,
    DatasetMoons,
    DatasetSiggraph,
    SyntheticDataset,
    DatasetkappaGMM,
)

ToyDatasetName = Literal["moons", "mixture", "siggraph", "checkerboard", "invertocat","kappagmm"]

TOY_DATASETS: dict[str, type[SyntheticDataset]] = {
    "moons": DatasetMoons,
    "mixture": DatasetMixture,
    "siggraph": DatasetSiggraph,
    "checkerboard": DatasetCheckerboard,
    "invertocat": DatasetInvertocat,
    "kappagmm" : DatasetkappaGMM,
}
