from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from cmaes import CMA
from scipy.stats import linregress
from tqdm import tqdm
from wirenec.scattering import get_scattering_in_frequency_range

from wirenec_optimization.parametrization.base_parametrization import (
    BaseStructureParametrization,
)


def objective_function(
    parametrization: BaseStructureParametrization,
    params: np.ndarray,
    freq: [list, tuple, np.ndarray] = tuple([10_000]),
    geometry: bool = False,
    scattering_angle: float = 90,
):
    g = parametrization.get_geometry(params=params)
    if not geometry:
        scattering, _ = get_scattering_in_frequency_range(
            g, freq, 90, 90, 90, scattering_angle
        )
        return (-1) * np.mean(scattering)
    else:
        return g


def check_convergence(
    progress, num_for_progress: int = 100, slope_for_progress: float = 1e-8
):
    if len(progress) > num_for_progress:
        slope1 = linregress(
            range(len(progress[-num_for_progress:])), progress[-num_for_progress:]
        ).slope
        slope2 = linregress(range(len(progress[-3:])), progress[-3:]).slope

        if abs(slope1) <= slope_for_progress and abs(slope2) <= slope_for_progress:
            print("Minimum slope converged")
            return True

    return False


def cma_optimize(
    structure_parametrization: BaseStructureParametrization,
    iterations: int = 200,
    seed: int = 48,
    frequencies: Tuple = tuple([9_000]),
    plot_progress: bool = False,
    scattering_angle: float = 90,
    population_size_factor: float = 1,
):
    np.random.seed(seed)
    bounds = structure_parametrization.bounds
    lower_bounds, upper_bounds = bounds[:, 0], bounds[:, 1]
    mean = lower_bounds + (np.random.rand(len(bounds)) * (upper_bounds - lower_bounds))
    sigma = 2 * (upper_bounds[0] - lower_bounds[0]) / 3

    optimizer = CMA(
        mean=mean,
        sigma=sigma,
        bounds=bounds,
        seed=seed,
        population_size=int(len(bounds) * population_size_factor),
    )

    cnt = 0
    max_value, max_params = 0, []

    pbar = tqdm(range(iterations))
    progress = []

    for generation in pbar:
        solutions = []
        values = []
        for _ in range(optimizer.population_size):
            params = optimizer.ask()

            value = objective_function(
                structure_parametrization,
                params,
                freq=frequencies,
                scattering_angle=scattering_angle,
            )
            values.append(value)
            if abs(value) > max_value:
                max_value = abs(value)
                max_params = params
                cnt += 1

            solutions.append((params, value))

        progress.append(-np.around(np.mean(values), 15))
        if check_convergence(progress):
            break

        pbar.set_description(
            "Processed %s generation\t max %s mean %s"
            % (generation, np.around(max_value, 15), -np.around(np.mean(values), 15))
        )

        optimizer.tell(solutions)

    if plot_progress:
        plt.plot(progress, marker=".", linestyle=":")
        plt.show()

    results = {
        "params": max_params,
        "optimized_value": -max_value,
        "progress": progress,
    }
    return results
