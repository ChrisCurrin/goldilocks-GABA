# -*- coding: utf-8 -*-
# -*- mode: python -*-
# Adapted from mpl_toolkits.axes_grid1
# LICENSE: Python Software Foundation (http://docs.python.org/license.html)

from matplotlib.offsetbox import AnchoredOffsetbox


class AnchoredScaleBar(AnchoredOffsetbox):
    def __init__(
        self,
        transform,
        sizex=0,
        sizey=0,
        labelx=None,
        labely=None,
        loc=4,
        pad=0.1,
        borderpad=0.1,
        sep=2,
        prop=None,
        textprops=None,
        barcolor="black",
        barwidth=None,
        y_rotation=90,
        **kwargs
    ):
        """
        Draw a horizontal and/or vertical  bar with the size in data coordinate
        of the give axes. A label will be drawn underneath (center-aligned).

        - transform : the coordinate frame (typically axes.transData)
        - sizex,sizey : width of x,y bar, in data units. 0 to omit
        - labelx,labely : labels for x,y bars; None to omit
        - loc : position in containing axes
        - pad, borderpad : padding, in fraction of the legend font size (or prop)
        - sep : separation between labels and bars in points.
        - **kwargs : additional arguments passed to base class constructor
        """
        from matplotlib.patches import Rectangle
        from matplotlib.offsetbox import AuxTransformBox, VPacker, HPacker, TextArea

        bars = AuxTransformBox(transform)
        if sizex:
            bars.add_artist(
                Rectangle((0, 0), sizex, 0, ec=barcolor, lw=barwidth, fc="none")
            )
        if sizey:
            bars.add_artist(
                Rectangle((0, 0), 0, sizey, ec=barcolor, lw=barwidth, fc="none")
            )

        if textprops is None:
            textprops = dict(ha="left")
        elif "ha" not in textprops:
            textprops["ha"] = "left"
        y_rotation = textprops.pop("y_rotation", y_rotation)

        if sizex and labelx:
            self.xlabel = TextArea(labelx, textprops=textprops)
            bars = VPacker(children=[bars, self.xlabel], align="center", pad=0, sep=sep)
        if sizey and labely:
            textprops["rotation"] = y_rotation
            self.ylabel = TextArea(labely, textprops=textprops)
            bars = HPacker(children=[self.ylabel, bars], align="center", pad=0, sep=sep)

        AnchoredOffsetbox.__init__(
            self,
            loc,
            pad=pad,
            borderpad=borderpad,
            child=bars,
            prop=prop,
            frameon=False,
            **kwargs
        )


def add_scalebar(
    ax,
    matchx=True,
    matchy=True,
    hidex=True,
    hidey=True,
    labelypad=0,
    fmt=".2f",
    **kwargs
):
    """Add scalebars to axes

    Adds a set of scale bars to *ax*, matching the size to the ticks of the plot
    and optionally hiding the x and y axes

    - ax : the axis to attach ticks to
    - matchx,matchy : if True, set size of scale bars to spacing between ticks
                    if False, size should be set using sizex and sizey params
    - hidex,hidey : if True, hide x-axis and y-axis of parent
    - **kwargs : additional arguments passed to AnchoredScaleBars

    Returns created scalebar object
    """

    def f(axis):
        tick_locs = axis.get_majorticklocs()
        return len(tick_locs) > 1 and (tick_locs[1] - tick_locs[0])

    if matchx or "sizex" not in kwargs:
        kwargs["sizex"] = f(ax.xaxis)

    if matchy or "sizey" not in kwargs:
        kwargs["sizey"] = f(ax.yaxis)

    if ":" not in fmt:
        fmt = ":" + fmt
    if "{" not in fmt:
        fmt = "{" + fmt + "}"
    size_x = fmt.format(kwargs["sizex"])
    size_y = fmt.format(kwargs["sizey"])

    if "labelx" in kwargs:
        kwargs["labelx"] = size_x + " " + kwargs["labelx"]
    else:
        kwargs["labelx"] = size_x

    if "labely" in kwargs:
        kwargs["labely"] = size_y + " " + kwargs["labely"]
    else:
        kwargs["labely"] = size_y

    if labelypad > 0:
        kwargs["labely"] = "\n" * labelypad + kwargs["labely"]
    else:
        kwargs["labely"] = kwargs["labely"] + "\n" * (-1 * labelypad)

    sb = AnchoredScaleBar(ax.transData, bbox_transform=ax.transAxes, **kwargs)
    ax.add_artist(sb)

    if hidex:
        ax.xaxis.set_visible(False)
    if hidey:
        ax.yaxis.set_visible(False)
    if hidex and hidey:
        ax.set_frame_on(False)

    return sb
