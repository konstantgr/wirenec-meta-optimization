import numpy as np

from wirenec.geometry import Wire, Geometry
from wirenec.visualization import plot_geometry
from wirenec.geometry.samples import double_srr_6GHz

from .base_parametrization import BaseObjectParametrization


def get_geometry_dimensions(geom: Geometry):
    wires = geom.wires
    points = [w.p1 for w in wires] + [w.p2 for w in wires]
    points = np.array(points).T
    mx = 0
    for dim in points:
        mx = max(mx, dim.max() - dim.min())
    return mx


class WireParametrization(BaseObjectParametrization):
    def __init__(self, max_size: float = 20 * 1e-3, min_size: float = 2 * 1e-3):
        super().__init__("Wire", max_size, min_size)

    def get_geometry(self, size_ratio, orientation, wire_radius: float = 0.5 * 1e-4):
        length = self.min_size + (self.max_size - self.min_size) * size_ratio
        g = Geometry([Wire(
            (0, -length / 2, 0),
            (0, length / 2, 0), wire_radius)])
        g.rotate(*orientation)
        return g
class SSRRParametrization(BaseObjectParametrization):
    def __init__(self, max_size: float = 3.25 * 1e-3, min_size: float = 9 * 1e-3):
        super().__init__("SSRR", max_size, min_size)

    def get_geometry(self, size_ratio, orientation, wire_radius: float = 0.5 * 1e-4):
        length = self.min_size + (self.max_size - self.min_size) * size_ratio
        gap = (length)/2
        dist = (length)/8
        g = Geometry([Wire((gap/2, length/2, 0.), (length/2, length/2, 0.), wire_radius),
                      Wire((length/2, length/2, 0.), (length/2, -length/2, 0.), wire_radius),
                      Wire((length/2, -length/2, 0.), (-length/2, -length/2, 0.), wire_radius),
                      Wire((-length/2, -length/2, 0.), (-length/2, length/2, 0.), wire_radius),
                      Wire((-length/2, length/2, 0.), (-gap/2, length/2, 0.), wire_radius),

                      Wire((length/2 - dist, length / 2 - dist, 0.), (length / 2 - dist, - length / 2 + dist, 0.), wire_radius),
                      Wire((length / 2 - dist, - length / 2 + dist, 0.), (gap / 2 - dist, -length / 2 + dist, 0.), wire_radius),
                      Wire((- gap / 2 + dist, -length / 2 + dist, 0.), (-length / 2 + dist, -length / 2 + dist, 0.), wire_radius),
                      Wire((-length / 2 + dist, -length / 2 + dist, 0.), (-length / 2 + dist, length / 2 - dist, 0.), wire_radius),
                      Wire((-length / 2 + dist, length / 2 - dist, 0.), (length/2 - dist, length / 2 - dist, 0.), wire_radius)
                      ])
        g.rotate(*orientation)
        return g
# class SSRRParametrization(BaseObjectParametrization):
#     def __init__(self, max_size: float = 3.25 * 1e-3, min_size: float = 9 * 1e-3):
#         super().__init__("SSRR", max_size, min_size)
#
#     def get_geometry(self, size_ratio, orientation, wire_radius: float = 0.5 * 1e-4):
#         length = self.min_size + (self.max_size - self.min_size) * size_ratio
#         gap = length/2
#         dist = length/10
#         wires = []
#         for layer in range(1):
#             parity = 1 if layer % 2 else -1
#             steps = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
#
#             side_length = length - 2 * dist * layer
#             coords = np.array([
#                 (gap / 2, side_length / 2, 0.),
#                 *[(x_factor * side_length / 2, y_factor * side_length / 2, 0) for (x_factor, y_factor) in steps],
#                 (-gap / 2, side_length / 2, 0.)
#             ]).astype(float) * parity
#         for c1, c2 in zip(coords, coords[1:]):
#             coords_extended = np.linspace(c1, c2, endpoint=True)
#             wires += [Wire(p1, p2, radius=wire_radius) for p1, p2 in zip(coords_extended, coords_extended[1:])]
#         g = Geometry(wires)
#         g.rotate(*orientation)
#         return g

# class SRRParametrization(BaseObjectParametrization):
#     def __init__(self, max_size: float = 3.25 * 1e-6, min_size: float = 9 * 1e-6):
#         super().__init__("SSRR", max_size, min_size)
#     def get_geometry(self, size_ratio, orientation, wire_radius: float = 0.5 * 1e-7):
#         r = self.min_size + (self.max_size - self.min_size) * size_ratio
#         g = double_srr_6GHz(r=r)
#         g.rotate(*orientation)
#         return g


if __name__ == '__main__':
    wire_param = WireParametrization(20 * 1e-4)
    srr_param = SSRRParametrization()

    g = srr_param.get_geometry(1, (0, 0, 0))
    plot_geometry(g)
