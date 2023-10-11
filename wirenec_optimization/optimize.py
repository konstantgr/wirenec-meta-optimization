import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from wirenec.geometry import Geometry, Wire
from wirenec.scattering import get_scattering_in_frequency_range
from wirenec.visualization import plot_geometry, scattering_plot

from wirenec_optimization.export_utils.utils import get_macros
from wirenec_optimization.optimization_utils.cmaes_optimizer import cma_optimizer, objective_function
from wirenec_optimization.parametrization.layers_parametrization import LayersParametrization
from wirenec_optimization.parametrization.spatial_parametrization import SpatialParametrization


def dipolar_limit(freq):
    c = 299_792_458
    lbd = c / (freq * 1e6)
    lengths = lbd / 2

    res = []
    for i, l in enumerate(lengths):
        g = Geometry([Wire((0, 0, -l / 2), (0, 0, l / 2), 0.5 * 1e-3)])
        f = freq[i]
        scattering = get_scattering_in_frequency_range(g, [f], 90, 90, 0, 270)
        res.append(scattering[0][0])

    return freq, np.array(res)


def write_to_file(file_path, content, write_mode='w+', is_json=True):
    with open(file_path, write_mode) as fp:
        if is_json:
            json.dump(content, fp)
        else:
            if write_mode == 'wb':
                np.save(fp, np.array(content))
            else:
                fp.write(content)


def save_results(
        parametrization,
        param_hyperparams: dict,
        opt_hyperparams: dict,
        optimized_dict: dict,
        path: str = "data/optimization/",
):
    path += f'{parametrization.structure_name}__'
    for param, value in param_hyperparams.items():
        path += f"{param}_{str(value)}__"
    for param, value in opt_hyperparams.items():
        path += f"{param}_{str(value)}__"
    path = Path(path.rstrip("_"))

    path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(2, figsize=(6, 8))

    g_optimized = objective_function(parametrization, params=optimized_dict['params'], geometry=True)
    _, backward_scattering = scattering_plot(
        ax[0], g_optimized, eta=90, num_points=100,
        scattering_phi_angle=90,
        label='Optimized Geometry. Backward'
    )

    freq, forward_scattering = scattering_plot(
        ax[0], g_optimized, eta=90, num_points=100,
        scattering_phi_angle=270,
        label='Optimized Geometry. Forward'
    )

    final_spectra_stats = {
        'backward_argmax': freq[np.argmax(backward_scattering)],
        'forward_argmax': freq[np.argmax(forward_scattering)],
        'backward_max': np.max(backward_scattering),
        'forward_max': np.max(forward_scattering)
    }

    x, y = dipolar_limit(np.linspace(5_000, 14_000, 100))

    parameters_count = int(parametrization.optimized_objects_count)
    ax[0].plot(x, np.array(y) * parameters_count, color='b', linestyle='--', label=f'{parameters_count} Bound')
    ax[0].plot(x, np.array(y), color='k', linestyle='--', label=f'Single dipole bound')

    ax[0].set_xlim(5_000, 14_000)

    ax[1].plot(optimized_dict['progress'], marker='.', linestyle=':')
    ax[0].legend()

    fig.savefig(path / 'scattering_progress.pdf', dpi=200, bbox_inches='tight')
    plt.show()

    plot_geometry(g_optimized, from_top=False, save_to=path / 'optimized_geometry.pdf')

    optimized_dict['params'] = optimized_dict['params'].tolist()

    write_to_file(f'{path}/parametrization_hyperparams.json', param_hyperparams)
    write_to_file(f'{path}/optimization_hyperparams.json', opt_hyperparams)
    write_to_file(f'{path}/optimized_params.json', optimized_dict)
    write_to_file(f'{path}/progress.npy', optimized_dict['progress'], 'wb', False)
    write_to_file(f'{path}/macros.txt', get_macros(g_optimized), 'w', False)
    write_to_file(f'{path}/optimized_results.json', final_spectra_stats)


if __name__ == "__main__":
    # TODO Optimize main
    layers_parametrization_hyperparams = {
        'matrix_size': (3, 3), 'layers_num': 2,
        'tau': 20 * 1e-3, 'delta': 10 * 1e-3,
        'asymmetry_factor': 0.9
    }

    spatial_parametrization_hyperparams = {
        'matrix_size': (2, 1, 1),
        'tau_x': 20 * 1e-3,
        'tau_y': 20 * 1e-3,
        'tau_z': 20 * 1e-3,
        'asymmetry_factor': 0.9
    }

    multiple_seeds = False
    optimization_type = 'spatial'

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
            'iterations': 5, 'seed': 42,
            "frequencies": tuple([10_000]), "scattering_angle": 90
        }

        optimized_dict = cma_optimizer(parametrization, **optimization_hyperparams)
        save_results(parametrization, parametrization_hyperparams, optimization_hyperparams, optimized_dict)
    else:

        custom_path = f"data/optimization/experiment_{time.strftime('%I:%M%p_%B_%d_%Y')}/"

        for seed in tqdm(range(0, 5)):
            optimization_hyperparams = {
                'iterations': 5, 'seed': seed,
                "frequencies": tuple([10_000]), "scattering_angle": 90
            }
            optimized_dict = cma_optimizer(parametrization, **optimization_hyperparams)
            save_results(
                parametrization,
                parametrization_hyperparams,
                optimization_hyperparams,
                optimized_dict,
                path=custom_path
            )