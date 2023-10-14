import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from wirenec.visualization import plot_geometry

from wirenec_optimization.export_utils.utils import get_macros
from wirenec_optimization.optimization_utils.cmaes_optimizer import (
    cma_optimizer,
    objective_function,
)
from wirenec_optimization.parametrization.layers_parametrization import (
    LayersParametrization,
)
from wirenec_optimization.parametrization.spatial_parametrization import (
    SpatialParametrization,
)
from wirenec_optimization.results_processing import (
    plot_optimization_progress,
    plot_optimized_scattering,
    write_to_file,
)


def save_results(
    parametrization,
    param_hyperparams: dict,
    opt_hyperparams: dict,
    optimized_dict: dict,
    path: str = "data/optimization/",
):
    path += f"{parametrization.structure_name}__"
    for param, value in param_hyperparams.items():
        path += f"{param}_{str(value)}__"
    for param, value in opt_hyperparams.items():
        path += f"{param}_{str(value)}__"
    path = Path(path.rstrip("_"))

    path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(2, figsize=(6, 8))

    tmp = plot_optimized_scattering(
        parametrization, objective_function, optimized_dict, ax[0]
    )
    g_optimized, freq, backward_scattering, forward_scattering, ax[0] = tmp

    ax[1] = plot_optimization_progress(optimized_dict, ax[1])

    final_spectra_stats = {
        "backward_argmax": freq[np.argmax(backward_scattering)],
        "forward_argmax": freq[np.argmax(forward_scattering)],
        "backward_max": np.max(backward_scattering),
        "forward_max": np.max(forward_scattering),
    }

    fig.savefig(path / "scattering_progress.pdf", dpi=200, bbox_inches="tight")
    plt.show()

    plot_geometry(g_optimized, from_top=False, save_to=path / "optimized_geometry.pdf")

    optimized_dict["params"] = optimized_dict["params"].tolist()

    write_to_file(f"{path}/parametrization_hyperparams.json", param_hyperparams)
    write_to_file(f"{path}/optimization_hyperparams.json", opt_hyperparams)
    write_to_file(f"{path}/optimized_params.json", optimized_dict)
    write_to_file(f"{path}/progress.npy", optimized_dict["progress"], "wb", False)
    write_to_file(f"{path}/macros.txt", get_macros(g_optimized), "w", False)
    write_to_file(f"{path}/optimized_results.json", final_spectra_stats)


if __name__ == "__main__":
    layers_parametrization_hyperparams = {
        "matrix_size": (2, 2),
        "layers_num": 2,
        "tau": 20 * 1e-3,
        "delta": 10 * 1e-3,
        "asymmetry_factor": 0.9,
    }

    spatial_parametrization_hyperparams = {
        "matrix_size": (2, 1, 1),
        "tau_x": 20 * 1e-3,
        "tau_y": 20 * 1e-3,
        "tau_z": 20 * 1e-3,
        "asymmetry_factor": 0.9,
    }

    multiple_seeds = False
    optimization_type = "layers"

    if optimization_type == "layers":
        parametrization_hyperparams = layers_parametrization_hyperparams
        parametrization = LayersParametrization(**parametrization_hyperparams)
    elif optimization_type == "spatial":
        parametrization_hyperparams = spatial_parametrization_hyperparams
        parametrization = SpatialParametrization(**parametrization_hyperparams)
    else:
        raise Exception("Unknown optimization type")

    if not multiple_seeds:
        optimization_hyperparams = {
            "iterations": 200,
            "seed": 42,
            "frequencies": tuple([10_000]),
            "scattering_angle": 90,
        }

        optimized_dict = cma_optimizer(parametrization, **optimization_hyperparams)
        save_results(
            parametrization,
            parametrization_hyperparams,
            optimization_hyperparams,
            optimized_dict,
        )
    else:
        custom_path = (
            f"data/optimization/experiment_{time.strftime('%I:%M%p_%B_%d_%Y')}/"
        )

        for seed in tqdm(range(0, 2)):
            optimization_hyperparams = {
                "iterations": 300,
                "seed": seed,
                "frequencies": tuple([10_000]),
                "scattering_angle": 90,
            }
            optimized_dict = cma_optimizer(parametrization, **optimization_hyperparams)
            save_results(
                parametrization,
                parametrization_hyperparams,
                optimization_hyperparams,
                optimized_dict,
                path=custom_path,
            )
