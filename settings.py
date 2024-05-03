# coding=utf-8
import logging
import os

import colorlog
import matplotlib.pyplot as plt
from brian2.units import second
from style.color import COLOR  # noqa
from style import text  # noqa
from core.var_ranges import *  # noqa

# ----------------------------------------------------------------------------------------------------------------------
# GLOBAL DEFAULTS
# ----------------------------------------------------------------------------------------------------------------------
time_unit = second
RASTERIZED = False  # True for faster rendering/saving
SAVE_FIGURES = True


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
