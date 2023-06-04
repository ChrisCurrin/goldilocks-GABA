# coding=utf-8
import logging
import os

import colorlog
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from brian2.units import second
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import (
    FuncNorm,
    LinearSegmentedColormap,
    LogNorm,
    Normalize,
    TwoSlopeNorm,
)

from style import constants  # noqa

# ----------------------------------------------------------------------------------------------------------------------
# GLOBAL DEFAULTS
# ----------------------------------------------------------------------------------------------------------------------
time_unit = second
RASTERIZED = False  # True for faster rendering/saving
SAVE_FIGURES = True

_fp = np.log2(15) % 1  # just get the fractional part
# 2**3.9069 == 15
# 2**7.9069 == 240
# 2**8.9069 == 480
TAU_KCC2_LIST = [
    int(g) for g in np.logspace(start=3 + _fp, stop=7 + _fp, base=2, num=9)
]
# TAU_KCC2_LIST = [int(g) for g in np.logspace(start=3+_fp, stop=8+_fp, base=2, num=11)]
G_GABA_LIST = [25, 50, 100, 200]

# ----------------------------------------------------------------------------------------------------------------------
# SET LOGGER
# ----------------------------------------------------------------------------------------------------------------------
handler = colorlog.StreamHandler()
fmt = "%(asctime)s [%(levelname)8s] %(message)-90s (%(name)s::%(filename)s::%(lineno)s)"
datefmt = "%Y-%m-%d %H:%M:%S"
handler.setFormatter(colorlog.ColoredFormatter("%(log_color)s" + fmt, datefmt=datefmt))


logging.basicConfig(
    level=logging.getLevelName(os.getenv("LOG_LEVEL", "INFO")),
    format=fmt,
    handlers=[handler],
    datefmt=datefmt,
)
modules_to_ignore = [
    "matplotlib",
    "colormath.color_conversions",
    "asyncio",
    "fontTools",
]
for module in modules_to_ignore:
    logging.getLogger(module).setLevel(logging.WARNING)

# ----------------------------------------------------------------------------------------------------------------------
# MATPLOTLIB PLOT CONFIG
# ----------------------------------------------------------------------------------------------------------------------
article_style_path = "style/article.mplstyle"
if not os.path.isfile(article_style_path):
    # find the file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    article_style_path = os.path.join(dir_path, article_style_path)
plt.style.use(article_style_path)
logging.getLogger("settings").debug("imported style {}".format(article_style_path))

# DEFINE FIGURE USEFUL SIZES (in inches)
PAGE_W_FULL = 7.5
PAGE_H_FULL = 7.5  # make square so there's space for caption
PAGE_H_FULL_no_cap = 8.75  # no space for caption
PAGE_W_half = PAGE_W_FULL / 2
PAGE_H_half = PAGE_H_FULL_no_cap / 2
PAGE_W_3rd = PAGE_W_FULL / 3
PAGE_H_3rd = PAGE_H_FULL_no_cap / 3
PAGE_W_4th = PAGE_W_FULL / 4
PAGE_H_4th = PAGE_H_FULL_no_cap / 4
PAGE_W_column = (
    5.2  # according to https://journals.plos.org/ploscompbiol/s/figures#loc-dimensions
)
# GridSpec layout
GS_R = 36
GS_C = 36
GS_R_half = int(GS_R / 2)
GS_C_half = int(GS_C / 2)
GS_R_third = int(GS_R / 3)
GS_C_third = int(GS_C / 3)
GS_R_4th = int(GS_R / 4)
GS_C_4th = int(GS_C / 4)
grid_spec_size = (GS_R, GS_C)
HPAD = 4
WPAD = 4

# DEFINE SOME HELPFUL COLOR CONSTANTS
default_colors = [
    "1f77b4",
    "ff7f0e",
    "2ca02c",
    "d62728",
    "9467bd",
    "8c564b",
    "e377c2",
    "7f7f7f",
    "bcbd22",
    "17becf",
]


def blend(c1="#d62728", c2="#1f77b4", ratio=0.5):
    """Shorthand for most common form of mixing colors

    see also: `sns.blend_palette`

    >>> COLOR.blend(COLOR.E, COLOR.I, 0.5).hexcode
    '#7b4f6e'
    """
    from spectra import html

    return html(c1).blend(html(c2), ratio)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    if type(cmap) is str:
        cmap = plt.get_cmap(cmap)
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    >> lighten_color((.3,.55,.1), 2.)
    """
    import colorsys

    import matplotlib.colors as mc

    try:
        c = mc.cnames[color]
    except (KeyError, TypeError):
        c = color
    hue, lum, sat = colorsys.rgb_to_hls(*mc.to_rgb(c))
    lum = 1 - amount * (1 - lum)
    lum = min(1, lum)
    lum = max(0, lum)
    return colorsys.hls_to_rgb(hue, lum, sat)


def categorical_cmap(nc: int, nsc: int, cmap="tab10", continuous=False):
    """
    You may use the HSV system to obtain differently saturated and luminated colors for the same hue. Suppose you
    have at most 10 categories, then the tab10 map can be used to get a certain number of base colors. From those you
    can choose a couple of lighter shades for the subcategories.

    https://stackoverflow.com/questions/47222585/matplotlib-generic-colormap-from-tab10

    :param nc: number of categories
    :param nsc: number of subcategories
    :param cmap: base colormap
    :param continuous: smooth out subcategory colors
    :return: colormap with nc*nsc different colors, where for each category there are nsc colors of same hue
    """
    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0, 1, nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc * nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv, nsc).reshape(nsc, 3)
        arhsv[:, 1] = np.linspace(chsv[1], 0.25, nsc)
        arhsv[:, 2] = np.linspace(chsv[2], 1, nsc)
        rgb = colors.hsv_to_rgb(arhsv)
        _start = i * nsc
        _end = (i + 1) * nsc
        cols[_start:_end, :] = rgb
    cmap = colors.ListedColormap(cols)
    return cmap


class G_GABA_Norm(TwoSlopeNorm, LogNorm):
    """
    Log-normalise data with a center.
    """

    pass


class COLOR:
    """COLOR object for consistent choice of COLORS wherever settings.py is used"""

    B = "#1f77b4"
    O = "#ff7f0e"  # noqa: E741
    G = "#2ca02c"
    R = "#d62728"
    Pu = "#9467bd"
    Br = "#8c564b"
    Pi = "#e377c2"
    K = "#7f7f7f"
    Ye = "#bcbd22"
    Cy = "#17becf"
    R1_B2 = "#552f72"
    R1_B3 = "#403580"
    R2_B1 = "#802456"
    R3_B1 = "#bf122b"
    E_I = "#7b4f6e"  # 50-50 mix
    # assign semantic colors
    exc = R
    inh = B
    average = K
    exc_alt = O
    NMDA = exc_alt
    AMPA = exc
    GABA = inh
    C_E_E = Br
    C_I_E = Ye
    C_E_I = Pu
    C_I_I = Cy

    blockGABA = Pi
    benzoGABA = "#98fb98"

    EIA_list = [exc, inh, K]
    g_list = [NMDA, AMPA, GABA]
    CONN_list = [C_E_E, C_I_E, C_E_I, C_I_I]

    g_dict = dict(NMDA=NMDA, AMPA=AMPA, GABA=GABA)
    RATE_dict = dict(r_E=exc, r_I=inh, r_all=average)
    CONN_dict = dict(
        C_E_E=C_E_E,
        C_I_E=C_I_E,
        C_E_I=C_E_I,
        C_I_I=C_I_I,
        synapse_mon_cee=C_E_E,
        synapse_mon_cie=C_I_E,
        synapse_mon_cei=C_E_I,
        synapse_mon_cii=C_I_I,
    )

    CONN_BLEND = dict(E=R3_B1, I=R1_B3)

    # to get the appropriate EGABA color in range [-80, -40] call EGABA_SM.to_rgba(<egaba value>)
    EGABA_SM = ScalarMappable(norm=Normalize(-80, -40), cmap="Blues_r")
    EGABA_2_SM = ScalarMappable(
        norm=TwoSlopeNorm(vmin=-74, vcenter=-60, vmax=-42), cmap="coolwarm"
    )
    G_AMPA_SM = ScalarMappable(norm=Normalize(0, 20), cmap="Reds_r")

    # TAU_PAL = sns.color_palette("hls", n_colors=len(TAU_KCC2_LIST), desat=1.)
    TAU_PAL = sns.color_palette(
        [
            "#29AF8C",
            # "#97BE49",
            # "#4CC756",
            "#73CC33",
            "#3D9CCC",
            "#7C60C6",
            "#D62466",
            "#D58C2E",
            "#C9492C",
            "#44546A",
            "#21406B",
        ]
    )
    TAU_PAL_DICT = dict(zip(TAU_KCC2_LIST, TAU_PAL))
    TAU_SM = ScalarMappable(
        norm=LogNorm(1, max(TAU_KCC2_LIST)),
        cmap=LinearSegmentedColormap.from_list("TAU_cm", TAU_PAL),
    )

    G_GABA_PAL = sns.blend_palette(
        [blockGABA, K, benzoGABA], n_colors=len([1] + G_GABA_LIST)
    )
    G_GABA_PAL_DICT = dict(zip([1] + G_GABA_LIST, G_GABA_PAL))
    G_GABA_SM = ScalarMappable(
        norm=G_GABA_Norm(50, 1, 1000),
        cmap=sns.blend_palette(
            [blockGABA] * 5 + [K] + [benzoGABA] * 5,
            n_colors=11,
            as_cmap=True,
        ),
    )

    blend = staticmethod(blend)
    truncate_colormap = staticmethod(truncate_colormap)
    lighten_color = staticmethod(lighten_color)
    categorical_cmap = staticmethod(categorical_cmap)

    @staticmethod
    def get_egaba_color(egaba):
        """Helper method to get the color value for a given EGABA.

        Range is [-88, -40]"""
        from brian2 import Quantity, mV

        if type(egaba) is Quantity:
            egaba = egaba / mV
        return COLOR.EGABA_SM.to_rgba(egaba)


benzo_map = {"default": COLOR.average}
_picro_N = np.round(np.linspace(0, 1, 100, endpoint=False), 2)
for i in _picro_N:
    benzo_map[i] = lighten_color(COLOR.Pi, 1 - i + 0.3 * i)  # [1, 0.3]
_benzo_N = np.linspace(1, 10, 19)
for i in _benzo_N:
    benzo_map[i] = lighten_color(COLOR.benzoGABA, 1 + i * 2 / 10)  # [1,2]
benzo_map[1] = benzo_map["default"]

# Have colormaps separated into categories:
# http://matplotlib.org/examples/color/colormaps_reference.html
cmaps = {
    "Perceptually Uniform Sequential": ["viridis", "plasma", "inferno", "magma"],
    "Sequential": [
        "Greys",
        "Purples",
        "Blues",
        "Greens",
        "Oranges",
        "Reds",
        "YlOrBr",
        "YlOrRd",
        "OrRd",
        "PuRd",
        "RdPu",
        "BuPu",
        "GnBu",
        "PuBu",
        "YlGnBu",
        "PuBuGn",
        "BuGn",
        "YlGn",
    ],
    "Sequential (2)": [
        "binary",
        "gist_yarg",
        "gist_gray",
        "gray",
        "bone",
        "pink",
        "spring",
        "summer",
        "autumn",
        "winter",
        "cool",
        "Wistia",
        "hot",
        "afmhot",
        "gist_heat",
        "copper",
    ],
    "Diverging": [
        "PiYG",
        "PRGn",
        "BrBG",
        "PuOr",
        "RdGy",
        "RdBu",
        "RdYlBu",
        "RdYlGn",
        "Spectral",
        "coolwarm",
        "bwr",
        "seismic",
    ],
    "Qualitative": [
        "Pastel1",
        "Pastel2",
        "Paired",
        "Accent",
        "Dark2",
        "Set1",
        "Set2",
        "Set3",
        "tab10",
        "tab20",
        "tab20b",
        "tab20c",
    ],
    "Miscellaneous": [
        "flag",
        "prism",
        "ocean",
        "gist_earth",
        "terrain",
        "gist_stern",
        "gnuplot",
        "gnuplot2",
        "CMRmap",
        "cubehelix",
        "brg",
        "hsv",
        "gist_rainbow",
        "rainbow",
        "jet",
        "nipy_spectral",
        "gist_ncar",
    ],
}
