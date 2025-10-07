# pyright: reportUnknownMemberType=false
from pathlib import Path
from typing import Unpack

import numpy as np
from matplotlib import pyplot as plt
from pytools.plotting.api import create_figure, style_kwargs, update_figure_setting
from pytools.plotting.trait import PlotKwargs

from .struct import PreppedData, Segmentation


def plot_prepped_data[F: np.floating](
    data: PreppedData[F], fout: Path, **kwargs: Unpack[PlotKwargs]
) -> None:
    defaults: PlotKwargs = {"figsize": (8, 6), "padleft": 0.12, "padbottom": 0.3, "linewidth": 0.75}
    kwargs = defaults | kwargs
    fig, ax = create_figure(nrows=4, **kwargs)
    update_figure_setting(fig, **kwargs)
    ax_style = style_kwargs(**kwargs)
    ax[0].plot(data.x, **ax_style)
    ax[1].plot(data.y, **ax_style)
    ax[2].plot(data.dy / data.dy.max(), **ax_style)
    ax[3].plot(data.ddy / data.ddy.max(), **ax_style)
    ax[0].set_ylabel("Raw Data")
    ax[1].set_ylabel("Smoothed Data")
    ax[2].set_ylabel("Derivative")
    ax[3].set_ylabel("Second Derivative")
    ax[3].set_xlabel("Time")
    fig.savefig(fout)
    plt.close(fig)


def plot_segmentation_part[F: np.floating, I: np.integer](
    data: PreppedData[F],
    segmentation: Segmentation[F, I],
    part: str,
    fout: Path,
    **kwargs: Unpack[PlotKwargs],
) -> None:
    defaults: PlotKwargs = {"figsize": (8, 2), "padleft": 0.12, "padbottom": 0.3, "linewidth": 0.75}
    kwargs = defaults | kwargs
    fig, ax = create_figure(**kwargs)
    update_figure_setting(fig, **kwargs)
    ax_style = style_kwargs(**kwargs)
    if part not in segmentation.prot:
        msg = f"Protocol '{part}' not found in segmentation."
        raise ValueError(msg)
    segments = [i for cycle in segmentation.prot[part].values() for i in cycle]
    segments = [min(segments) - 1, *segments]
    start = max(segmentation.idx[min(segments)], 0)
    end = (
        min(segmentation.idx[min(max(segments) + 1, segmentation.n - 1)], segmentation.idx.max())
        + 1
    )
    ax.plot(np.arange(start, end, dtype=int), data.x[start:end], "k-", label="Data", **ax_style)
    ax.plot(
        segmentation.idx[segments],
        data.x[segmentation.idx[segments]],
        "ro",
        label="Splits",
        **ax_style,
    )
    ax.set_ylabel("Segmentation")
    ax.set_xlabel("Time")
    ax.set_yticks([])
    ax.legend(title="Test", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.savefig(fout.parent / (fout.stem + f"_{part}.png"))
    plt.close(fig)
