import numpy as np
from wirenec.geometry import Geometry
from wirenec.visualization import plot_geometry

from wirenec_optimization.parametrization.base_parametrization import BaseStructureParametrization
from wirenec_optimization.parametrization.sample_objects import WireParametrization, SRRParametrization


class ConformalParametrization(BaseStructureParametrization):
    def __init__(self, matrix_size, tau_x, tau_y, tau_z):
        super().__init__("layers")

        self.type_mapping = {
            0: WireParametrization,
            1: SRRParametrization
        }

    @property
    def bounds(self) -> np.ndarray:
       pass

    def get_random_geometry(self, seed: int = 42) -> Geometry:
        np.random.seed(seed)
        bounds = self.bounds
        random_parameters = [np.random.uniform(low=mn, high=mx) for (mn, mx) in bounds]
        return self.get_geometry(random_parameters)

    def get_geometry(self, params: [np.ndarray, list]) -> Geometry:
        pass


import numpy as np

def data_for_cylinder_along_z(center_x,center_y,radius,height_z):
    z = np.linspace(-height_z/2, height_z/2, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid



import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch, Rectangle
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d


if __name__ == '__main__':
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.use('macosx')

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    circle = Rectangle((0, 0), 1, 1, alpha=0.5)
    ax.add_patch(circle)
    art3d.pathpatch_2d_to_3d(circle, z=0, zdir='y')

    phi = np.linspace(0, 2*np.pi, 100)
    r = 0.7
    W = 1
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    y = np.zeros_like(x)

    ax.scatter(x, np.zeros_like(x), y)


    r = 2 / (2*np.pi)

    phi_new = (x + 1)/2*2*np.pi
    z_new = y
    x_new = r * np.cos(phi_new)
    y_new = r * np.sin(phi_new)

    ax.scatter(y_new, x_new, z_new, color='r')

    Xc, Yc, Zc = data_for_cylinder_along_z(0, 0, r, 2)
    ax.plot_surface(Xc, Yc, Zc, alpha=0.5)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    plt.show()
    # param = SpatialParametrization(
    #     matrix_size=(3, 3, 3),
    #     tau_x=20 * 1e-3,
    #     tau_y=20 * 1e-3,
    #     tau_z=20 * 1e-3,
    # )
    # g = param.get_random_geometry(seed=42)
    # plot_geometry(g)
