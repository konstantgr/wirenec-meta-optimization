import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from wirenec.geometry import Geometry, Wire
from wirenec.scattering import get_scattering_in_frequency_range
from wirenec.visualization import scattering_plot  # , plot_geometry

from export_utils.utils import get_macros
from optimization_utils.cmaes_optimizer import cma_optimizer, objective_function
from parametrization.layers_parametrization import LayersParametrization
from wirenec_optimization.optimization_utils.visualization import plot_geometry


def dipolar_limit(freq):
    c = 299_792_458
    lbd = c / (freq * 1e6)
    lengths = lbd / 2

    res = []
    for i, l, in enumerate(lengths):
        g = Geometry([Wire((0, 0, -l / 2), (0, 0, l / 2), 0.5 * 1e-3)])
        f = freq[i]
        scattering = get_scattering_in_frequency_range(g, [f], 90, 90, 0, 270)
        res.append(scattering[0][0])

    return freq, np.array(res)


#scattering_angle = (80, 90, 100)
scattering_angle = (170, 180, 190)
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

    scatter_1 = scattering_plot(
        ax[0], g_optimized, theta=scattering_angle[0], eta=0, num_points=100,
        scattering_phi_angle=scattering_angle[0],
        label='Scattering angle:' + ' ' + str(scattering_angle[0]) + '$\degree$'
    )

    scatter_2 = scattering_plot(
        ax[0],  g_optimized, theta=scattering_angle[1], eta=0, num_points=100,
        scattering_phi_angle=scattering_angle[1],
        label='Scattering angle:' + ' ' + str(scattering_angle[1]) + '$\degree$'
    )

    scatter_3 = scattering_plot(
        ax[0], g_optimized, theta=scattering_angle[2], eta=0, num_points=100,
        scattering_phi_angle=scattering_angle[2],
        label='Scattering angle:' + ' ' + str(scattering_angle[2]) + '$\degree$'
    )

    x, y = dipolar_limit(np.linspace(2_000, 10_000, 100))

    parameters_count = (
        int(len(optimized_dict['params']) / 5)  # two more parameters for deltas
        if param_hyperparams["asymmetry_factor"] is not None
        else int(len(optimized_dict['params']) / 3)
    )
    # ax[0].plot(x, np.array(y) * parameters_count, color='b', linestyle='--', label=f'{parameters_count} Bound')
    # ax[0].plot(x, np.array(y), color='k', linestyle='--', label=f'Single dipole bound')

    ax[0].set_xlim(2_000, 10_000)

    ax[1].plot(optimized_dict['progress'], marker='.', linestyle=':')
    ax[0].legend()

    fig.savefig(path / 'scattering_progress.pdf', dpi=200, bbox_inches='tight')
    plt.show()

    plot_geometry(g_optimized, from_top=False, save_to=path / 'optimized_geometry.pdf')

    with open(f'data/Wire_3_1e2_Opt.txt', "w+") as file:
            file.write('freq' + '\t' + 'scaterring_' + str(scattering_angle[0]) + '\t'
                       + 'scaterring_' + str(scattering_angle[1])
                       + '\t' + 'scaterring_' + str(scattering_angle[2]) + '\n')
            for i in range(len(scatter_1[0])):
                file.write(str(scatter_1[0][i]) + '\t' + str(scatter_1[1][i]) + '\t'
                           + str(scatter_2[1][i]) + '\t' + str(scatter_3[1][i]) + '\n')

    with open(f'{path}/parametrization_hyperparams.json', 'w+') as fp:
        json.dump(param_hyperparams, fp)
    with open(f'{path}/optimization_hyperparams.json', 'w+') as fp:
        json.dump(opt_hyperparams, fp)
    with open(f'{path}/optimized_params.json', 'w+') as fp:
        optimized_dict['params'] = optimized_dict['params'].tolist()
        json.dump(optimized_dict, fp)
    with open(f'{path}/progress.npy', 'wb') as fp:
        np.save(fp, np.array(optimized_dict['progress']))
    with open(f'{path}/macros.txt', 'w+') as fp:
        fp.write(get_macros(g_optimized))


if __name__ == "__main__":
    parametrization_hyperparams = {
        'matrix_size': (3, 3), 'layers_num': 1,
        'tau': 20 * 1e-3, 'delta': 10 * 1e-3,
        'asymmetry_factor': None
    }

    # parametrization_hyperparams = {
    #     'matrix_size': (2, 2, 2),
    #     'tau_x': 20 * 1e-3,
    #     'tau_y': 20 * 1e-3,
    #     'tau_z': 20 * 1e-3,
    # }

    optimization_hyperparams = {
        'iterations': 1, 'seed': 42,
        "frequencies": tuple([10000]), "scattering_angle": scattering_angle
    }

    parametrization = LayersParametrization(**parametrization_hyperparams)
    # parametrization = SpatialParametrization(**parametrization_hyperparams)
    optimized_dict = cma_optimizer(parametrization, **optimization_hyperparams) 
    save_results(parametrization, parametrization_hyperparams, optimization_hyperparams, optimized_dict)
