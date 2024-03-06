from functools import partial

import brian2.numpy_ as np
from ipywidgets import HBox, SelectMultiple, VBox, interactive

from widgets.common import (
    layout_width,
    manual_dict,
    manual_run_button_style,
    style,
    style_desc_width_auto,
)

#############################
# Excitatory Synapses Params
#############################

gGABAs_all = [
    0,
    25,
    50,
    100,
    200,
]
gGABAs_selected = [
    25,
    50,
    100,
]
gAMPAs = sorted(np.round(np.arange(0, 20.0001, 5.0), 0))
gNMDAs = [5.0, 7.5, 10.0]
seeds = (None, 1013, 12987, 1234, 1837)
egabas_plot = [-42, -56, -70]

exc_widget_part = partial(
    interactive,
    gGABAs=SelectMultiple(
        options=gGABAs_all,
        value=gGABAs_selected,
        rows=len(gGABAs_selected),
        description="gGABAs",
        style=style,
        layout=layout_width,
    ),
    gAMPAs=SelectMultiple(
        options=gAMPAs,
        value=gAMPAs,
        rows=len(gAMPAs),
        description="gAMPAs",
        style=style,
        layout=layout_width,
    ),
    gNMDAs=SelectMultiple(
        options=gNMDAs,
        value=gNMDAs,
        rows=len(gNMDAs),
        description="gNMDAs",
        style=style,
        layout=layout_width,
    ),
    seeds=SelectMultiple(
        options=seeds,
        value=seeds,
        rows=len(seeds),
        description="Seeds",
        style=style,
        layout=layout_width,
    ),
    egabas_plot=SelectMultiple(
        options=sorted(np.arange(-74, -38, 2)),
        value=egabas_plot,
        rows=10,
        description="EGABAs to plot",
        style=style_desc_width_auto,
    ),
)


def new_exc_widget(f):
    exc_widget = exc_widget_part(f, manual_dict)

    exc_widget.children[-2].button_style = manual_run_button_style

    # put plotting on the right
    exc_widget.children = (
        HBox(
            [
                *exc_widget.children[:5],
            ],
        ),
        VBox(exc_widget.children[5:]),
    )

    return exc_widget
