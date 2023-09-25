from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, TwoSlopeNorm

from core.var_ranges import G_GABA_LIST, TAU_KCC2_LIST


def opacity(level, color):
    if level > 1:
        level /= 100
    _opacity = "%0.2X" % round(
        (level * 255)
    )  # note: not sure if round or int works better here
    if len(color) == 9:
        # already has an opacity applied, so reset it by removing last 2
        color = color[:-2]
    return color + _opacity


def get_benzo_color(drug: float, ampa: float = None) -> Tuple[str, Union[str, None]]:
    """Get color for drug based.

    :param: drug to match (according to `settings.benzo_map`).
    :return: (benzo strength, ampa strength [None if not specified])
    """

    ampa_color = None if ampa is None else COLOR.G_AMPA_SM.to_rgba(ampa)
    return COLOR.G_GABA_SM.to_rgba(drug * 50), ampa_color


def get_drug_label(
    drug: float, decimals=0, ampa: float = None
) -> Tuple[str, Union[str, None]]:
    return f"{np.round(drug, decimals)}", f"{ampa:.0f}" if ampa else None


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

    # blockGABA = "#ca01a9"  # pink
    # blockGABA = "#BE894A"  # gold
    blockGABA = "#e94e1b"  # gold
    benzoGABA = "#00c6ff"  # blue
    # benzoGABA = "#98fb98" # green

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

    ECL = ecl = "#00bf71"  # green
    EGABA = "#312783"  # dark blue
    # to get the appropriate EGABA color in range [-80, -40] call EGABA_SM.to_rgba(<egaba value>)
    EGABA_reverse = True  # this is for plots that may need this info
    EGABA_SM = ScalarMappable(
        norm=Normalize(-80, -40),
        cmap=sns.light_palette(EGABA, as_cmap=True, reverse=EGABA_reverse),  # greens
    )
    EGABA_2_SM = ScalarMappable(
        norm=TwoSlopeNorm(vmin=-74, vcenter=-60, vmax=-42), cmap="coolwarm"
    )
    G_AMPA_SM = ScalarMappable(norm=Normalize(0, 20), cmap="Reds_r")

    KCC2 = "#a3195b"
    NKCC1 = "#e94e1b"
    # TAU_PAL = sns.blend_palette(
    #     # ["#B33F1B", "#a3195b", "#6D12B0"], n_colors=len(TAU_KCC2_LIST)
    #     # ["#C1DB14", "#DC8C14", "#EB1528"], n_colors=len(TAU_KCC2_LIST)
    #     # ["#00BD1D", "#7DAE00", "#BA9A09"], n_colors=len(TAU_KCC2_LIST),
    #     # ["#a3195b", "#7DAE00", "#EFD6AC"],
    #     # n_colors=len(TAU_KCC2_LIST),
    # )
    # TAU_PAL = sns.color_palette([K]*len(TAU_KCC2_LIST), n_colors=len(TAU_KCC2_LIST))
    # TAU_PAL = sns.dark_palette(K, n_colors=len(TAU_KCC2_LIST))
    TAU_PAL = sns.dark_palette(ECL, n_colors=len(TAU_KCC2_LIST) + 2, reverse=True)[
        : len(TAU_KCC2_LIST)
    ]
    TAU_PAL_DICT = dict(zip(TAU_KCC2_LIST, TAU_PAL))
    TAU_SM = ScalarMappable(
        norm=LogNorm(min(TAU_KCC2_LIST), max(TAU_KCC2_LIST)),
        cmap=LinearSegmentedColormap.from_list("TAU_cm", TAU_PAL),
    )

    G_GABA_PAL = sns.blend_palette(
        [blockGABA, K, benzoGABA], n_colors=len([1] + G_GABA_LIST)
    )
    G_GABA_PAL_DICT = dict(zip([1] + G_GABA_LIST, G_GABA_PAL))
    _num_up_down = 20
    G_GABA_SM = ScalarMappable(
        norm=G_GABA_Norm(50, 1, 1000),
        cmap=sns.blend_palette(
            sns.blend_palette(
                [
                    lighten_color(blockGABA, 1.2),
                    blockGABA,
                    lighten_color(blockGABA, 0.8),
                ],
                _num_up_down,
            )
            + [K]
            + sns.blend_palette([benzoGABA, "#18596C", "#0c2e38"], _num_up_down),
            n_colors=_num_up_down * 2 + 1,
            as_cmap=True,
        ),
    )

    NUM_BURSTS_COLOR = "#000000"
    NUM_BURSTS_CMAP = sns.color_palette(
        "Greys", as_cmap=True
    )  # sns.blend_palette(["#FFE8F1", "#F00A80"], as_cmap=True)
    NUM_BURSTS_PALETTE = sns.color_palette(
        "Greys", 8
    )  # sns.blend_palette(["#FFE8F1", "#F00A80"], 8, as_cmap=False)

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
    benzo_map[i] = lighten_color(COLOR.blockGABA, 1 - i + 0.3 * i)  # [1, 0.3]
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
