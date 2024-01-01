from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import ray
from cmaes import CMA
from ray.util.multiprocessing import Pool
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
    scattering_angle: tuple = (90,),
    maximize: bool = False,
):
    g = parametrization.get_geometry(params=params)
    factor = -1 if maximize else 1
    if not geometry:
        scattering, _ = get_scattering_in_frequency_range(
            g, freq, 90, 90, 90, scattering_angle
        )
        return factor * np.mean(scattering)

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
    scattering_angle: tuple = (90,),
    population_size_factor: float = 1,
    maximize: bool = False,
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
    best_value, best_params = 0 if maximize else np.inf, []

    progress = []

    ray.init(num_cpus=8)
    pool = Pool()

    for generation in tqdm(range(iterations)):
        solutions = []
        params_list = [optimizer.ask() for _ in range(optimizer.population_size)]

        with tqdm(total=optimizer.population_size) as pbar:
            values = []
            for value in pool.imap(
                lambda params: objective_function(
                    structure_parametrization,
                    params,
                    freq=frequencies,
                    scattering_angle=scattering_angle,
                    maximize=maximize,
                ),
                params_list,
            ):
                values.append(value)
                pbar.update()

        for params, value in zip(params_list, values):
            condition = value > best_value if maximize else value < best_value
            if condition:
                best_value = value
                best_params = params
                cnt += 1

            solutions.append((params, value))

        progress.append(-np.around(np.mean(values), 15))
        if check_convergence(progress):
            break

        pbar.set_description(
            "Processed %s generation\t max %s mean %s"
            % (generation, np.around(best_value, 15), -np.around(np.mean(values), 15))
        )

        optimizer.tell(solutions)

    pool.close()
    pool.join()
    ray.shutdown()

    if plot_progress:
        plt.plot(progress, marker=".", linestyle=":")
        plt.show()

    results = {
        "params": best_params,
        "optimized_value": -best_value,
        "progress": progress,
    }
    return results
