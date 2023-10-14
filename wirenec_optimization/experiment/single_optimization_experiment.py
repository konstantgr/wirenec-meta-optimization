from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from wirenec.visualization import plot_geometry

from wirenec_optimization.experiment import (
    plot_optimized_scattering,
    plot_optimization_progress,
    write_to_file,
)
from wirenec_optimization.experiment.base_experiment import BaseExperiment
from wirenec_optimization.export_utils.utils import get_macros
from wirenec_optimization.optimization_utils.cmaes_optimizer import (
    objective_function,
    cma_optimize,
)
from wirenec_optimization.parametrization.layers_parametrization import (
    LayersParametrization,
)
from wirenec_optimization.parametrization.spatial_parametrization import (
    SpatialParametrization,
)

parametrization_mapping = {
    "layers": LayersParametrization,
    "spatial": SpatialParametrization,
}


class SingleOptimizationExperiment(BaseExperiment):
    def __init__(self, config: DictConfig):
        self.config = config
        self.parametrization_name = config.get("parametrization_name")
        self.parametrization_hyperparams = OmegaConf.to_container(
            config.get("parametrization_hyperparams")
        )
        self.optimization_hyperparams = OmegaConf.to_container(
            config.get("optimization_hyperparams")
        )

        self.parametrization = parametrization_mapping.get(self.parametrization_name)(
            **self.parametrization_hyperparams
        )
        self.optimized_dict = None

    def run(self):
        self.optimized_dict = cma_optimize(
            self.parametrization, **self.optimization_hyperparams
        )

    @property
    def results(self):
        raise NotImplementedError

    def save_results(
        self,
        path: str = "data/optimization/",
    ) -> Any:
        path += f"{self.parametrization.structure_name}__"
        for param, value in self.parametrization_hyperparams.items():
            path += f"{param}_{str(value)}__"
        for param, value in self.optimization_hyperparams.items():
            path += f"{param}_{str(value)}__"
        path = Path(path.rstrip("_"))

        path.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(2, figsize=(6, 8))

        tmp = plot_optimized_scattering(
            self.parametrization, objective_function, self.optimized_dict, ax[0]
        )
        g_optimized, freq, backward_scattering, forward_scattering, ax[0] = tmp

        ax[1] = plot_optimization_progress(self.optimized_dict, ax[1])

        final_spectra_stats = {
            "backward_argmax": freq[np.argmax(backward_scattering)],
            "forward_argmax": freq[np.argmax(forward_scattering)],
            "backward_max": np.max(backward_scattering),
            "forward_max": np.max(forward_scattering),
        }

        fig.savefig(path / "scattering_progress.pdf", dpi=200, bbox_inches="tight")
        plt.show()

        plot_geometry(
            g_optimized, from_top=False, save_to=path / "optimized_geometry.pdf"
        )

        self.optimized_dict["params"] = list(self.optimized_dict["params"])

        write_to_file(
            f"{path}/parametrization_hyperparams.json", self.parametrization_hyperparams
        )
        write_to_file(
            f"{path}/optimization_hyperparams.json", self.optimization_hyperparams
        )
        write_to_file(f"{path}/optimized_params.json", self.optimized_dict)
        write_to_file(
            f"{path}/progress.npy", self.optimized_dict["progress"], "wb", False
        )
        write_to_file(f"{path}/macros.txt", get_macros(g_optimized), "w", False)
        write_to_file(f"{path}/optimized_results.json", final_spectra_stats)
