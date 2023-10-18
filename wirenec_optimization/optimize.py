from omegaconf import OmegaConf

from wirenec_optimization.experiment import (
    MultiSeedOptimizationExperiment,
    SingleOptimizationExperiment,
)

if __name__ == "__main__":
    config = OmegaConf.load("configs/single_layers_experiment.yaml")
    # config = OmegaConf.load("configs/single_spatial_experiment.yaml")
    # config = OmegaConf.load("configs/multi_seed_experiment.yaml")

    experiment = SingleOptimizationExperiment(config)
    # experiment = MultiSeedOptimizationExperiment(config)

    experiment.run()
    experiment.save_results()

    save_each_iteration = True
    # experiment.run(save_each_iteration=save_each_iteration)

    if not save_each_iteration:
        experiment.save_results()
