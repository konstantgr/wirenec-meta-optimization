import json
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from wirenec.geometry import Geometry, Wire
from wirenec.scattering import get_scattering_in_frequency_range
from wirenec.visualization import scattering_plot


def dipolar_limit(freq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def single_channel_limit(freq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c_const = 3 * 1e8
    f = np.array(freq)
    wvs = c_const / (f * 1e6)

    return freq, 3 * wvs**2 / (2 * np.pi)


def write_to_file(file_path, content, write_mode="w+", is_json=True):
    with open(file_path, write_mode) as fp:
        if is_json:
            json.dump(content, fp)
        else:
            if write_mode == "wb":
                np.save(fp, np.array(content))
            else:
                fp.write(content)


def plot_optimization_progress(optimized_dict: dict, ax: plt.axes) -> plt.axes:
    parameters = {"marker": ".", "linestyle": ":"}
    ax.plot(optimized_dict["progress"], **parameters)
    return ax


def plot_optimized_scattering(
    parametrization,
    objective_function: Callable,
    optimized_dict: dict,
    ax=plt.axes,
    freq_min: float = 5_000,
    freq_max: float = 14_000,
    num: int = 100,
    polarization_angle: float = 90.0,
    scattering_phi_angle: tuple = (90, 270),
    limit: Callable = dipolar_limit,
):
    # x, y = limit(np.linspace(freq_min, freq_max, num))
    parameters_count = int(parametrization.optimized_objects_count)
    # ax.plot(
    #     x,
    #     np.array(y) * parameters_count,
    #     color="b",
    #     linestyle="--",
    #     label=f"{parameters_count} Bound",
    # )
    # ax.plot(x, np.array(y), color="k", linestyle="--", label="Single dipole bound")

    g_optimized = objective_function(
        parametrization, params=optimized_dict["params"], geometry=True
    )
    scattering_dict = {}
    for angle in scattering_phi_angle:
        freq, scattering = scattering_plot(
            ax,
            g_optimized,
            eta=polarization_angle,
            frequency_start=freq_min,
            frequency_finish=freq_max,
            num_points=num,
            phi=angle,
            scattering_phi_angle=angle,
            label=f"Optimized Geometry. {angle} degrees",
        )
        scattering_dict[angle] = scattering

    ax.set_xlim(freq_min, freq_max)
    ax.legend()

    return g_optimized, freq, scattering_dict, ax
