import re
import numpy as np

from wirenec.geometry import Wire, Geometry


def vba_wire(p1, p2, wire_radius, name):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    s = f'''With Wire
     .Reset 
     .Name "{name}" 
     .Folder "" 
     .Type "BondWire" 
     .Height "0" 
     .Radius "{wire_radius}" 
     .Point1 "{x1}", "{y1}", "{z1}", "False"
     .Point2 "{x2}", "{y2}", "{z2}", "False"
     .BondWireType "Spline" 
     .Alpha "75" 
     .Beta "35" 
     .RelativeCenterPosition "0.5" 
     .Material "PEC" 
     .SolidWireModel "False" 
     .Termination "Extended" 
     .Add
    End With'''
    return s


def get_macros(g, history=False):
    s = ''
    for i, wire in enumerate(g.wires):
        p1, p2 = wire.p1 * 1e3, wire.p2 * 1e3
        wire_radius = wire.radius * 1e3
        s += '\n' + vba_wire(p1, p2, wire_radius, str(i)) + '\n'

    if history:
        res = '\nDim t           As String\nt = ""\n'
        for line in s.splitlines():
            line = line.strip().replace('"', '""')
            res += f't = t & "{line}" & vbCrLf\n'

        res += 'AddToHistory("Created", t)\n'
        return res

    return s


def read_data_cst(path):
    with open(path, 'r') as f:
        tmp = f.readlines()

    d = []
    for line in tmp[4:-1]:
        d.append(line.strip().split())
    x, y = np.array(d).astype(float).T
    return x * 1000, y


def cst2nec(file_path):
    with open(file_path, 'r') as f:
        s = f.read()

    wires = []
    i = 0
    for w_s in s.split('\n\n'):
        for line in w_s.splitlines():
            if 'Point1' in line:
                p1 = np.round(np.array(re.findall(r"[-+]?(?:\d*\.*\d+)", line)).astype(float)[1:], 7) * 1e-3
            if 'Point2' in line:
                p2 = np.round(np.array(re.findall(r"[-+]?(?:\d*\.*\d+)", line)).astype(float)[1:], 7) * 1e-3
            if 'Radius' in line:
                rad = np.array(re.findall(r"[-+]?(?:\d*\.*\d+)", line)).astype(float)[0] * 1e-3

        x1, y1, _ = p1
        x2, y2, _ = p2

        i += 1
        segments = 3 if rad == 0.00025 else 10
        wires.append(Wire(
            p1, p2, rad, segments=segments
        ))

    g = Geometry(wires)
    return g
