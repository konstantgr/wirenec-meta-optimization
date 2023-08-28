import numpy as np
from wirenec.geometry import Geometry
from wirenec.visualization import plot_geometry

from wirenec_optimization.parametrization.base_parametrization import BaseStructureParametrization
from wirenec_optimization.parametrization.sample_objects import WireParametrization, SRRParametrization


class SpatialParametrization(BaseStructureParametrization):
    def __init__(self, matrix_size, tau_x, tau_y, tau_z):
        super().__init__("spatial")

        self.type_mapping = {
            0: WireParametrization,
            1: SRRParametrization
        }

        self.matrix_size = matrix_size

        self.tau_x = tau_x
        self.tau_y = tau_y
        self.tau_z = tau_z

    @property
    def bounds(self) -> np.ndarray:
        m, n, k = self.matrix_size

        size_bounds = [(0, 1) for _ in range(n * m)] * k
        orientation_bounds = [(0, 2 * np.pi) for _ in range(n * m)] * 3 * k
        types_bounds = [(0, len(self.type_mapping.keys()) - 1) for _ in range(n * m)] * k
        return np.array(types_bounds + size_bounds + orientation_bounds)

    def get_random_geometry(self, seed: int = 42) -> Geometry:
        np.random.seed(seed)
        bounds = self.bounds
        random_parameters = [np.random.uniform(low=mn, high=mx) for (mn, mx) in bounds]
        return self.get_geometry(random_parameters)

    def get_geometry(self, params: [np.ndarray, list]) -> Geometry:
        m, n, k = self.matrix_size
        split_size = m * n * k
        types_params, size_params, orientation_params = (
            np.array_split(params[:split_size], k),
            np.array_split(params[split_size:2 * split_size], k),
            np.array_split(params[2 * split_size:], k)
        )

        wires = []
        a_x, a_y, a_z = self.tau_x * n, self.tau_y * m, self.tau_z * k
        x0, y0, z0 = -a_x / 2 + self.tau_x / 2, -a_y / 2 + self.tau_y / 2, -a_y / 2 + self.tau_z / 2

        for l in range(k):
            for i in range(m):
                for j in range(n):
                    tp, size_ratio, orientation = (
                        int(np.around(types_params[l].reshape((m, n))[i, j])),
                        size_params[l].reshape((m, n))[i, j],
                        orientation_params[l].reshape((m, n, 3))[i, j]
                    )
                    orientation = tuple(orientation)
                    g_tmp = self.type_mapping[tp]().get_geometry(size_ratio, orientation)
                    x, y, z = x0 + self.tau_x * i, y0 + self.tau_y * j, self.tau_z * l
                    g_tmp.translate((x, y, z))
                    wires += g_tmp.wires

        return Geometry(wires)


if __name__ == '__main__':
    param = SpatialParametrization(
        matrix_size=(5, 5, 5),
        tau_x=20 * 1e-3,
        tau_y=20 * 1e-3,
        tau_z=20 * 1e-3,
    )
    g = param.get_random_geometry(seed=41)
    plot_geometry(g)
