import time

import numpy as np
from omegaconf import DictConfig, OmegaConf

from wirenec_optimization.experiment.single_optimization_experiment import (
    SingleOptimizationExperiment,
)
from wirenec_optimization.optimization_utils.cmaes_optimizer import cma_optimize


class MultiSeedOptimizationExperiment(SingleOptimizationExperiment):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.seeds = np.arange(*config.get("seeds_range"))
        self.optimization_results = {}

    def run(self):
        for seed in self.seeds:
            self.optimization_results[seed] = cma_optimize(
                self.parametrization, **self.optimization_hyperparams, seed=seed
            )

    def save_results(
        self,
        path: str = "data/optimization/",
    ):
        base_path = (
            f"data/optimization/experiment_{time.strftime('%I_%M_%p_%B_%d_%Y')}/"
        )
        for seed, optimized_dict in self.optimization_results.items():
            self.optimized_dict = optimized_dict
            self.optimization_hyperparams["seed"] = int(seed)
            super().save_results(base_path)


if __name__ == "__main__":
    conf = OmegaConf.load("../configs/multi_seed_experiment.yaml")
    exp = MultiSeedOptimizationExperiment(conf)
    print(exp.seeds)
