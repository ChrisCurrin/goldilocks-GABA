from typing import Tuple, Union

import numpy as np


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
    from settings import COLOR, benzo_map

    ampa_color = None if ampa is None else COLOR.G_AMPA_SM.to_rgba(ampa)
    return COLOR.G_GABA_SM.to_rgba(drug * 50), ampa_color
    if drug in benzo_map:
        return benzo_map[drug], ampa_color
    return benzo_map["default"], ampa_color


def get_drug_label(
    drug: float, decimals=0, ampa: float = None
) -> Tuple[str, Union[str, None]]:
    return f"{np.round(drug, decimals)}", f"{ampa:.0f}" if ampa else None
