from typing import Iterable
import dataclasses

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


@dataclasses.dataclass
class PlotData:

    title: str
    x_axis: np.ndarray
    y_axes: tuple[np.ndarray, ...]
    x_label: str
    y_label: tuple[str, ...]
    x_limit: tuple[float, float]
    y_limit: tuple[float, float]
    animated: bool
    show_accumulated: bool


def plot_animated(
    grid: tuple[int, int],
    data: tuple[PlotData, ...],
):
    assert len(data) == grid[0] * grid[1]

    if len(data) == 1:
        fig, axis = plt.subplots(*grid)
        axes = (axis,)
    else:
        fig, axes = plt.subplots(*grid)

    fig.tight_layout()

    def animate(i):
        print(i)
        plots = zip(axes, data)
        plot_data: PlotData
        for ax, plot_data in plots:
            ax.cla()
            ax.set_xlim(plot_data.x_limit)
            ax.set_ylim(plot_data.y_limit)
            ax.title.set_text(plot_data.title)
            ax.grid()
            for y, label in zip(plot_data.y_axes, plot_data.y_label):
                if plot_data.show_accumulated:
                    ax.plot(plot_data.x_axis[:i], y[:i], label=label)
                else:
                    ax.plot(plot_data.x_axis[i], y[i], label=label)
                    ax.plot(plot_data.x_axis[i], y[i], "o")
                    # ax.set_aspect("equal", "box")
                    ax.set_aspect("equal")
            ax.legend()
        return

    # anim = animation.FuncAnimation(fig, animate, frames=99, interval=50)
    return fig, animate
