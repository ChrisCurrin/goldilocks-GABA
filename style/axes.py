""""""
import logging
from string import ascii_letters

import numpy as np
from matplotlib import lines as mlines
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import (
    inset_axes,
    mark_inset,
    zoomed_inset_axes,
)

logger = logging.getLogger(__name__)

num_map = [
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
]


def num2roman(num):
    """convert a number to roman numerals (uppercase)
    from: https://stackoverflow.com/questions/28777219/basic-program-to-convert-integer-to-roman-numerals#28777781
    """
    roman = ""

    while num > 0:
        for i, r in num_map:
            while num >= i:
                roman += r
                num -= i

    return roman


def letter_axes(
    *ax_list,
    start="A",
    subscript=None,
    repeat_subscript=False,
    xy=(0.0, 1.0),
    xycoords="axes fraction",
    **kwargs,
):
    """Annotate a list of axes with uppercase letters for figure goodness.

    :param ax_list: Axes which to annotate.
    :type ax_list: Axes or List[Axes]
    :param start: The axis letter to start at. Can also be integer letter index, useful for subscripts: 3 for 'iii'
    :type start: str or int
    :param subscript: The letter to subscript (e.g. 'B') such that ax_list will be (i.e. Bi, Bii, ...)
    :type subscript: str
    :param repeat_subscript: Include the alphabet letter for every subscript. E.g. Ai, Aii, Aiii if `True` otherwise
    Ai, ii, iii if `False`
    :param xy: X and Y coordinate to place the letter. Relative to axes fraction, with bottom left
        (on data plot portion) being (0,0).
    :type xy: Tuple
    :param xycoords: The coordinate frame to use (`any valid value for Axes.annotate`)
    :param kwargs: passed to `Axes.annotate`
    """
    from matplotlib.cbook import flatten

    # get ax_list into a flatten List
    if type(ax_list[0]) is list:
        ax_list = ax_list[0]
    ax_list = list(flatten(ax_list))
    for i, ax in enumerate(ax_list):
        if type(ax) is str and start == "A":
            # letter passed as arg
            ax_list = ax_list[:i]
            start = ax
            break

    # determine if xy is a list of placements (and should iterate xy along with ax_list, zip-like)
    iter_xy = np.iterable(xy) and np.iterable(xy[0]) and len(xy) == len(ax_list)
    if subscript is not None and start == "A":
        start_idx = 1
    elif type(start) is int:
        start_idx = start
    else:
        start_idx = ascii_letters.find(start)

    if subscript is None:
        if (type(start) is str and len(start) != 1) or start_idx == -1:
            raise SyntaxError("'start' must be a single letter in the alphabet")
    else:
        if subscript not in ascii_letters or len(subscript) != 1:
            raise SyntaxError("'subscript' must be a single letter in the alphabet")

    for ax_n, ax in enumerate(ax_list):
        idx = ax_n + start_idx
        _letter = f"{num2roman(idx).lower()}" if subscript else ascii_letters[idx]
        if (subscript and repeat_subscript) or (subscript and idx == 1):
            _letter = f"{subscript}{_letter}"
        _xy = xy[ax_n] if iter_xy else xy
        ax.annotate(_letter, xy=_xy, xycoords=xycoords, fontsize="xx-large", **kwargs)


def use_scalebar(ax, **kwargs):
    """

    :type ax: np.ndarray or plt.Axes


    """
    from .scalebars import add_scalebar

    if type(ax) == list or type(ax) == np.ndarray:
        for sub_ax in ax:
            use_scalebar(sub_ax, **kwargs)
    sb = add_scalebar(ax, **kwargs)
    return sb


def adjust_spines(
    ax, spines, position=0, smart_bounds=False, sharedx=False, sharedy=False
):
    """Set custom visibility and positioning of of axes' spines.
    If part of a subplot with shared x or y axis, use `sharedx` or `sharedy` keywords, respectively.

    Noe: see seaborn's `despine`` method for a more modern and robust approach

    :param ax: Axes to adjust. Multidimensional lists and arrays may have unintended consequences.
    :type ax: Axes or List[Axes] or np.ndarray[Axes]
    :param spines: The list of spines to show and adjust by `position`.
    :type spines: List[str]
    :param position: Place the spine out from the data area by the specified number of points.
    :type position: float
    :param smart_bounds: Set the spine and associated axis to have smart bounds.
    :type smart_bounds: bool
    :param sharedx: True if part of a subplot with a shared x-axis. This keeps the x-axis visible, but will remove
        ticks for `ax` unless the correct spine is provided.
        If -1 is provided and ax is a List, then the last axis will have an x-axis.
    :type sharedx: bool
    :param sharedy: True if part of a subplot with a shared y-axis. This keeps the y-axis visible, but will remove
        ticks for `ax` unless the correct spine is provided.
        If -1 is provided and ax is a List, then the last axis will have a y-axis.
    :type sharedy: bool
    """
    if np.iterable(ax):
        ax = np.array(ax)
        for i, sub_axis in enumerate(ax):
            if sharedx or sharedy:
                from matplotlib.axes import SubplotBase

                if isinstance(sub_axis, SubplotBase):
                    sub_axis.label_outer()
                if sharedx >= 1 and i == ax.shape[0] - 1 and "bottom" not in spines:
                    spines.append("bottom")
                if (
                    sharedy >= 1
                    and (
                        (len(ax.shape) == 1 and i == ax.shape[0] - 1)
                        or (len(ax.shape) == 2 and i == ax.shape[1] - 1)
                    )
                    and "left" not in spines
                ):
                    spines.append("left")
            adjust_spines(
                sub_axis, spines, position, smart_bounds, sharedx=False, sharedy=False
            )
        return

    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(("outward", position))
            spine.set_visible(True)
        else:
            spine.set_visible(False)

    # turn off ticks where there is no spine
    if "left" in spines:
        ax.yaxis.set_visible(True)
        ax.yaxis.tick_left()
        ax.yaxis.set_label_position("left")
    elif "right" in spines:
        ax.yaxis.set_visible(True)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    else:
        # no yaxis ticks
        # if sharedy <= 0:
        #     ax.yaxis.set_visible(False)
        for label in ax.get_yticklabels(which="both"):
            label.set_visible(False)
        ax.tick_params(axis="y", which="both", left=False, right=False)

    if "bottom" in spines:
        ax.xaxis.set_visible(True)
        ax.xaxis.tick_bottom()
        ax.xaxis.set_label_position("bottom")
    elif "top" in spines:
        ax.xaxis.set_visible(True)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
    else:
        # no xaxis ticks
        # if sharedx <= 0:
        # ax.xaxis.set_visible(False)
        for label in ax.get_xticklabels(which="both"):
            label.set_visible(False)
        ax.tick_params(axis="x", which="both", bottom=False, top=False)


def colorbar_adjacent(
    mappable,
    position="right",
    size="2%",
    pad=0.05,
    orientation="vertical",
    ax=None,
    **kwargs,
):
    """Create colorbar using axes toolkit, but means axes cannot be realigned properly"""
    ax = ax or mappable.axes
    fig: Figure = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size, pad)
    return fig.colorbar(mappable, cax=cax, orientation=orientation, **kwargs)


def colorbar_inset(
    mappable,
    position="outer right",
    size="2%",
    orientation="vertical",
    ax=None,
    inset_axes_kwargs=None,
    **kwargs,
):
    """Create colorbar using axes toolkit by insetting the axis
    :param mappable:
    :type mappable: matplotlib.image.AxesImage
    :param position:
    :type position: str
    :param size:
    :type size: str
    :param orientation:
    :type orientation: str
    :param ax:
    :type ax: matplotlib.axes.Axes
    :param kwargs:
    :return: Color bar
    :rtype: matplotlib.colorbar.Colorbar
    """
    ax = ax or mappable.axes
    fig = ax.figure
    if inset_axes_kwargs is None:
        inset_axes_kwargs = {"borderpad": 0}
    if orientation == "vertical":
        height = "100%"
        width = size
    else:
        height = size
        width = "100%"
    if "outer" in position:
        # we use bbox to shift the colorbar across the entire image
        bbox = [0.0, 0.0, 1.0, 1.0]
        if "right" in position:
            loc = "center left"
            bbox[0] = 1.0
        elif "left" in position:
            loc = "center right"
            bbox[0] = 1.0
        elif "top" in position:
            loc = "lower left"
            bbox[1] = 1.0
            if orientation is None:
                orientation = "horizontal"
        elif "bottom" in position:
            loc = "upper left"
            bbox[1] = 1.0
            if orientation is None:
                orientation = "horizontal"
        else:
            raise ValueError(
                "unrecognised argument for 'position'. "
                "Valid locations are 'right' (default),'left','top', 'bottom' "
                "with each supporting 'inner' (default) and 'outer'"
            )
        ax_cbar = inset_axes(
            ax,
            width=width,
            height=height,
            loc=loc,
            bbox_to_anchor=bbox,
            bbox_transform=ax.transAxes,
            **inset_axes_kwargs,
        )
    else:
        ax_cbar = inset_axes(
            ax,
            width=width,
            height=height,
            loc=position.replace("inner", "").strip(),
            **inset_axes_kwargs,
        )
    return fig.colorbar(mappable, cax=ax_cbar, orientation=orientation, **kwargs)


def align_axes(unaligned_axes):
    """
    Change position of unaligned_axes such that they have the same (minimum) width regardless of colorbars
    Especially useful when sharing the x axis
    see https://stackoverflow.com/questions/46694889/matplotlib-sharex-with-colorbar-not-working
    :param unaligned_axes: List of Axes objects to be aligned
    :type unaligned_axes: List[Axes] or ndarray
    """
    # get minimum width
    pos = unaligned_axes[0].get_position()
    for ax in unaligned_axes[1:]:
        pos_check = ax.get_position()
        if pos_check.width < pos.width:
            pos = pos_check
    # realign
    for ax in unaligned_axes:
        pos2 = ax.get_position()
        ax.set_position([pos.x0, pos2.y0, pos.width, pos2.height])


def create_zoom(
    ax_to_zoom,
    inset_size,
    lines=None,
    loc="lower left",
    loc1=1,
    loc2=2,
    xlim=None,
    ylim=None,
    xticks=2,
    yticks=2,
    ec="C7",
    inset_kwargs=None,
    box_kwargs=None,
    connector_kwargs=None,
    **kwargs,
):
    """Zoom into an axis with an inset plot

    The valid locations (for `loc`, `loc1`, and `loc2`) are: 'upper right' : 1, 'upper left' : 2, 'lower left' : 3,
    'lower right' : 4,
    'right' : 5, 'center left' : 6, 'center right' : 7, 'lower center' : 8, 'upper center' : 9, 'center' : 10

    :param ax_to_zoom: Source axis which data will be copied from and inset axis inserted.
    :type ax_to_zoom: Axes
    :param inset_size: Zoom factor for `zoomed_inset_axes` if argument is a float, else is width and height,
        respectively, for `inset_axes`.
    :type inset_size: float or tuple
    :param lines: Lines to plot in inset axis. If None, all lines are plotted.
    :type lines: List[Line2D]
    :param loc: Location to place the inset axes.
    :param loc1: Corner to use for connecting the inset axes and the area in the
        parent axes. Pass 'all' to connect all corners (overrides loc2).
    :param loc2: Corner to use for connecting the inset axes and the area in the
        parent axes. Pass 'all' to connect all corners (overrides loc1).
    :param xlim: Limits of x-axis. Also limits data **plotted** when copying to the inset axis.
    :type xlim: float or Tuple[float, float]
    :param ylim: Limits of y-axis.
    :type ylim: float or Tuple[float, float]
    :param xticks: Number of ticks (int) or location of ticks (list) or no x-axis (False).
    :type xticks: int or list
    :param yticks: Number of ticks (int) or location of ticks (list) or no y-axis (False).
    :type yticks: int or list
    :param ec: Edge color for border of zoom and connecting lines
    :param inset_kwargs: Keywords for `inset_axes` or `zoomed_inset_axes`.
            E.g. dict(bbox_to_anchor=(1,1), bbox_transform=ax_to_zoom.transAxes)
    :type inset_kwargs: dict
    :param box_kwargs: Keywords for `mpl_toolkits.axes_grid1.inset_locator.mark_inset`.
        To remove not mark the inset axis at all, set box_kwargs to `'None'`
    :type box_kwargs: dict or str
    :param connector_kwargs: Keywords for connecting lines between inset axis and source axis.
        To remove connecting lines set to '`None'`.
    :type connector_kwargs: dict or str
    :param kwargs: Additional keyword arguments for plotting. See `Axes` keywords.

    :return: inset axis
    :rtype: Axes
    """
    if inset_kwargs is None:
        inset_kwargs = dict(bbox_to_anchor=None, bbox_transform=None)
    elif "bbox_to_anchor" in inset_kwargs:
        if inset_kwargs["bbox_to_anchor"] is None:
            inset_kwargs["bbox_transform"] = None
        elif "bbox_transform" not in inset_kwargs or (
            "bbox_transform" in inset_kwargs and inset_kwargs["bbox_transform"] is None
        ):
            inset_kwargs["bbox_transform"] = ax_to_zoom.transAxes
    if box_kwargs is None:
        box_kwargs = dict()
    if connector_kwargs is None:
        connector_kwargs = dict()

    axes_kwargs = dict(facecolor="white")

    if type(inset_size) is tuple:
        ax_inset: Axes = inset_axes(
            ax_to_zoom,
            width=inset_size[0],
            height=inset_size[1],
            loc=loc,
            axes_kwargs=axes_kwargs,
            **inset_kwargs,
        )
    else:
        ax_inset: Axes = zoomed_inset_axes(
            ax_to_zoom,
            zoom=inset_size,
            loc=loc,
            axes_kwargs=axes_kwargs,
            **inset_kwargs,
        )
    src = ax_to_zoom if lines is None else lines
    copy_lines(src, ax_inset, xlim=xlim, **kwargs)

    ax_inset.set_xlim(xlim)
    ax_inset.set_ylim(ylim)
    for _ticks, _ticks_axis in zip([xticks, yticks], ["x", "y"]):
        get_axis = ax_inset.get_xaxis if _ticks_axis == "x" else ax_inset.get_yaxis
        if _ticks:
            if type(_ticks) is int:
                from matplotlib.ticker import MaxNLocator

                get_axis().set_major_locator(MaxNLocator(_ticks))
            else:
                get_axis().set_ticks(_ticks)
        else:
            get_axis().set_visible(False)

    for spine in ax_inset.spines:
        ax_inset.spines[spine].set_visible(True)
        ax_inset.spines[spine].set_color(ec)
    ax_inset.tick_params(axis="both", which="both", color=ec, labelcolor=ec)
    if box_kwargs != "None":
        box_connectors = []
        if loc1 == "all" or loc2 == "all":
            loc1, loc2 = 1, 2
            box_patch, p1, p2 = mark_inset(
                ax_to_zoom, ax_inset, loc1=loc1, loc2=loc2, ec=ec, **box_kwargs
            )
            box_connectors.extend([p1, p2])
            loc1, loc2 = 3, 4

        box_patch, p1, p2 = mark_inset(
            ax_to_zoom, ax_inset, loc1=loc1, loc2=loc2, ec=ec, **box_kwargs
        )
        box_connectors.extend([p1, p2])
        for loc, spine in ax_inset.spines.items():
            spine.set(**box_kwargs)  # consistency between inset border and marked box
        if type(connector_kwargs) is dict:
            for bc in box_connectors:
                # put connectors on original axis instead of new axis so that they go behind new axis
                bc.remove()
                ax_to_zoom.add_patch(bc)
                bc.set_zorder(4)
                bc.set(**connector_kwargs)
        elif connector_kwargs == "None":
            for bc in box_connectors:
                bc.set(color="None")
    return ax_inset


# noinspection SpellCheckingInspection
line_props = {
    "agg_filter": "get_agg_filter",
    "alpha": "get_alpha",
    "antialiased": "get_antialiased",
    "color": "get_color",
    "dash_capstyle": "get_dash_capstyle",
    "dash_joinstyle": "get_dash_joinstyle",
    "drawstyle": "get_drawstyle",
    "fillstyle": "get_fillstyle",
    "label": "get_label",
    "linestyle": "get_linestyle",
    "linewidth": "get_linewidth",
    "marker": "get_marker",
    "markeredgecolor": "get_markeredgecolor",
    "markeredgewidth": "get_markeredgewidth",
    "markerfacecolor": "get_markerfacecolor",
    "markerfacecoloralt": "get_markerfacecoloralt",
    "markersize": "get_markersize",
    "markevery": "get_markevery",
    "rasterized": "get_rasterized",
    "solid_capstyle": "get_solid_capstyle",
    "solid_joinstyle": "get_solid_joinstyle",
    "zorder": "get_zorder",
}


def copy_lines(src, ax_dest, xlim=None, xunit=None, rel_lw=2.0, cmap=None, **kwargs):
    """Copies the lines from a source Axes `ax_src` to a destination Axes `ax_dest`.
    By default, linewidth is doubled. To disable, set rel_lw to 1 or provide 'lw' keyword.

    :param src: Source Axes or list of Line2D objects.
    :type src: Axes or list
    :param ax_dest: Destination Axes.
    :type ax_dest: Axes
    :param xlim: Set the domain of the axis. This will restrict what is plotted (hence should be accompanied by
        xunit), not just a call to `set_xlim`
    :type xlim: tuple or list or int
    :param xunit: The units used with xlim to restrict the domain.
    :type xunit: Quantity
    :param rel_lw: Relative linewidth change for each line in source. Default doubles the linewidth.
    :type rel_lw: float
    :param kwargs: Keywords to provide to `Axes.plot` that will overwrite source properties. The aliases to use are
        defined in `line_props`.
    """
    from matplotlib import cbook, colors

    kwargs = cbook.normalize_kwargs(kwargs, mlines.Line2D._alias_map)
    if isinstance(src, Axes):
        lines = src.get_lines()
    elif np.iterable(src):
        lines = src
    else:
        lines = [src]
    if xlim is not None and xunit is not None:
        if np.iterable(xlim):
            xlim = [int(xlim[0] / xunit), int(xlim[1] / xunit)]
        else:
            xlim = [int(xlim / xunit), -1]
    if cmap is not None and type(cmap) is str:
        cmap = plt.get_cmap(cmap)
    for i, line in enumerate(lines):
        props = {}
        for prop, getter in line_props.items():
            props[prop] = getattr(line, getter)()
        if cmap is not None:
            props["color"] = colors.to_hex(cmap(i), keep_alpha=False)
        props = {**props, **kwargs}
        if "linewidth" not in kwargs:
            # change relative thickness of line
            props["linewidth"] *= rel_lw
        x, y = line.get_data()
        if xlim is not None and xunit is not None:
            _x0, _x1 = xlim
            x = x[_x0:_x1]
            y = y[_x0:_x1]
        ax_dest.plot(x, y, **props)


def colorline(
    x,
    y,
    z=None,
    cmap="copper",
    norm=plt.Normalize(0.0, 1.0),
    linewidth=1.0,
    ax=None,
    **kwargs,
):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """
    import matplotlib.collections as mcoll

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    # to check for numerical input -- this is a hack
    if not hasattr(z, "__iter__"):
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(
        segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, **kwargs
    )

    if ax is None:
        ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments
