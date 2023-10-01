import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects


kind_colors = {
    None: 'b',
    'Wire': 'b',
    'SRR': 'r',
    'SSRR': 'g'
}


def plot_geometry(g, from_top=False, save_to=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_params = {
        "linewidth": 1,
        "alpha": 0.8,
        # "path_effects": [path_effects.SimpleLineShadow(), path_effects.Normal()]
    }

    if from_top:
        ax.view_init(elev=90, azim=270)

    wires = g.wires
    for wire in wires:
        p1, p2 = wire.p1, wire.p2
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        ax.plot([x1, x2], [y1, y2], [z1, z2],
                color=kind_colors[wire.kind],
                **plot_params)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.ticklabel_format(style='sci', scilimits=(0, 0))

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    if save_to is not None:
        plt.savefig(save_to, bbox_inches='tight', dpi=200)

    plt.show()
