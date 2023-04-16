import logging
from typing import Iterable, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from settings import GS_C, GS_R, PAGE_W_FULL, PAGE_H_half


def plot_save(
    path: Union[str, Iterable], figs: list = None, close=True, tight=False, **kwargs
) -> None:
    """Save figures to path (including filename and extension)

    If the extension is .pdf then all figures save to a single multipage PDF file.

    :param path: relative URI path including directory, filename, and extension
    :param figs: Optional list of figures to save (defaults to all of them)
    :param close: Close the figures after saving them (default: True)
    :param tight: call fig.tight_layout() before saving (default: False)
    :param kwargs: other keyword arguments passed to fig.savefig
    :raises IOError: if anything goes wrong (making directory, saving, closing, etc.)
    """
    logger = logging.getLogger("plot_save")
    if not isinstance(path, str) and np.iterable(path):
        if len(path) > 1:
            for i in range(len(path)):
                if i == 0:
                    # do tight_layout once
                    plot_save(path[i], figs, close=False, tight=tight)
                elif i == len(path) - 1:
                    # only close on last path
                    plot_save(path[i], figs, close=close, tight=False)
                else:
                    # neither close nor tight_layout otherwise
                    #   don't want to apply tight_layout multiple times
                    #   can't save a figure if it has already been closed
                    plot_save(path[i], figs, close=False, tight=False)
        else:
            plot_save(path[0], figs, close=close, tight=tight)
    else:
        try:
            import os

            from matplotlib.backends.backend_pdf import PdfPages

            directory = os.path.split(path)[0]
            if not os.path.exists(directory):
                os.makedirs(directory)
            i = 1
            tmp_path = path
            while os.path.exists(tmp_path) and os.path.isfile(tmp_path):
                tmp_path = path.replace(".", "_{}.".format(i))
                i += 1
            path = tmp_path
            if figs is None:
                figs = [plt.figure(n) for n in plt.get_fignums()]
            if path.endswith(".pdf"):
                pp = PdfPages(path)
            else:
                pp = path
            logger.info("saving to {}".format(path))
            from matplotlib.figure import Figure

            fig: Figure
            for f_i, fig in enumerate(figs):
                if tight:
                    fig.tight_layout()
                if path.endswith(".pdf"):
                    pp.savefig(fig, **kwargs)
                else:
                    dpi = kwargs.pop("dpi", 600)
                    if len(figs) > 1:
                        pp = path.replace(".", "_fignum{}.".format(f_i))
                    fig.savefig(pp, dpi=dpi, **kwargs)
            if path.endswith(".pdf"):
                pp.close()
            logger.info(
                "Saved figures [{}]".format(",".join([str(fig.number) for fig in figs]))
            )
            if close:
                for fig in figs:
                    plt.close(fig)
        except IOError as save_err:
            logger.error("Cannot save figures. Error: {}".format(save_err))
            raise save_err


def adjust_ylabels(ax, x_offset=0):
    """
    Scan all ax list and identify the outmost y-axis position.
    Setting all the labels to that position + x_offset.
    """

    xc = np.inf
    for a in ax:
        xc = min(xc, (a.yaxis.get_label()).get_position()[0])

    for a in ax:
        label = a.yaxis.get_label()
        t = label.get_transform()
        a.yaxis.set_label_coords(xc + x_offset, label.get_position()[1], transform=t)
        label.set_va("top")


def annotate_cols_rows(axes, cols=None, rows=None, row_pad=5):
    if rows is None:
        rows = []
    if cols is None:
        cols = []
    annotate_cols(axes, cols)
    annotate_rows(axes, rows, pad=row_pad)


def annotate_cols(axes, labels):
    """SET COLUMN TITLE"""
    if len(axes) > 0:
        for ax, col in zip(axes[0], labels):
            ax.set_title(col)


def annotate_rows(axes, labels, pad=5):
    """SET ROW TITLE"""
    if len(axes) > 0:
        for ax, row in zip(axes[:, 0], labels):
            ax.annotate(
                row,
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                size="large",
                ha="right",
                va="center",
            )


def new_gridspec(
    nrows=GS_R,
    ncols=GS_C,
    figsize=(PAGE_W_FULL, PAGE_H_half),
    fig_kwargs=None,
    grid_kwargs=None,
) -> Tuple[Figure, GridSpec]:
    """Figure setup"""
    fig_kwargs = fig_kwargs or dict()
    grid_kwargs = grid_kwargs or dict()
    fig = plt.figure(figsize=figsize, **fig_kwargs)
    gs = GridSpec(nrows, ncols, figure=fig, **grid_kwargs)
    return fig, gs
