import numpy as np
from wirenec.geometry import Wire, Geometry
from wirenec.geometry.samples import double_SRR
from wirenec.visualization import plot_geometry

from wirenec_optimization.parametrization.base_parametrization import (
    BaseObjectParametrization,
)


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

    def get_geometry(self, size_ratio, orientation, wire_radius: float = 0.5 * 1e-3):
        length = self.min_size + (self.max_size - self.min_size) * size_ratio
        g = Geometry([Wire((0, -length / 2, 0), (0, length / 2, 0), wire_radius)])
        g.rotate(*orientation)
        return g


def double_srr_updated(r=3.25 * 1e-3, p0=(0, 0, 0), wr=0.25 * 1e-3, num=20):
    g = double_SRR(
        inner_radius=r, outer_radius=r + 5 * wr, wire_radius=wr, num_of_wires=num
    )
    g.translate(p0)

    return g


class SRRParametrization(BaseObjectParametrization):
    def __init__(self, max_size: float = 9 * 1e-3, min_size: float = 3.5 * 1e-3):
        super().__init__("SRR", max_size, min_size)

    def get_geometry(self, size_ratio, orientation, wire_radius: float = 0.5 * 1e-3):
        r = self.min_size + (self.max_size - self.min_size) * size_ratio
        g = double_srr_updated(r=r, wr=wire_radius)
        g.rotate(*orientation)
        return g


if __name__ == "__main__":
    wire_param = WireParametrization(20 * 1e-3)
    srr_param = SRRParametrization(min_size=2.5 * 1e-3)

    g = srr_param.get_geometry(0, (0, 0, 0), wire_radius=0.1 * 1e-3)
    plot_geometry(g)
