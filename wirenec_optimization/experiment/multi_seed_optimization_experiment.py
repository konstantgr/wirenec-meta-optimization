import time

import numpy as np
from omegaconf import DictConfig, OmegaConf

from wirenec_optimization.experiment.base_experiment import BaseExperiment
from wirenec_optimization.experiment.single_optimization_experiment import (
    SingleOptimizationExperiment,
)


class MultiSeedOptimizationExperiment(BaseExperiment):
    @property
    def results(self):
        return 1

    def __init__(self, config: DictConfig):
        self.config = config
        self.seeds = np.arange(*config.get("seeds_range"))
        self.optimization_results = {}
        self.start_time_str = None

    def run(self, save_each_iteration: bool = True):
        self.start_time_str = time.strftime("%I_%M_%p_%B_%d_%Y")

        for seed in self.seeds:
            self.config.optimization_hyperparams.seed = int(seed)
            experiment = SingleOptimizationExperiment(self.config)
            experiment.run()

            if save_each_iteration:
                base_path = f"data/optimization/experiment_{self.start_time_str}/"
                experiment.save_results(base_path)

            self.optimization_results[seed] = experiment

    def save_results(
        self,
        path: str = "data/optimization/",
    ):
        base_path = f"data/optimization/experiment_{self.start_time_str}/"
        for experiment in self.optimization_results.values():
            experiment.save_results(base_path)


if __name__ == "__main__":
    conf = OmegaConf.load("../configs/multi_seed_experiment.yaml")
    exp = MultiSeedOptimizationExperiment(conf)
    print(exp.seeds)
