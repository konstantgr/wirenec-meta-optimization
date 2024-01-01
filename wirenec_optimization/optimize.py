from omegaconf import OmegaConf

from wirenec_optimization.experiment import (
    MultiSeedOptimizationExperiment,
    SingleOptimizationExperiment,
)


def run_single_seed_experiment(experiment_type: str):
    if experiment_type == "layers":
        config = OmegaConf.load("configs/single_layers_experiment.yaml")
    elif experiment_type == "spatial":
        config = OmegaConf.load("configs/single_spatial_experiment.yaml")
    else:
        raise Exception("Unknown type")

    experiment = SingleOptimizationExperiment(config)

    experiment.run()
    experiment.save_results()


def run_multi_seed_experiment(save_each_iteration: bool = True):
    config = OmegaConf.load("configs/multi_seed_experiment.yaml")
    experiment = MultiSeedOptimizationExperiment(config)

    experiment.run(save_each_iteration=save_each_iteration)

    if not save_each_iteration:
        experiment.save_results()


if __name__ == "__main__":
    run_multi_seed_experiment()
