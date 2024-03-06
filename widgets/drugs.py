from functools import partial

import brian2.numpy_ as np
from ipywidgets import (
    HTML,
    FloatSlider,
    HBox,
    IntSlider,
    Layout,
    SelectMultiple,
    VBox,
    interactive,
)

from widgets.common import layout_width, manual_dict, manual_run_button_style, style

############################
# Drugs Figure Widgets
############################

benzo_strengths = (0, 0.25, 0.5, 1, 2, 4, 8)
all_benzo_strengths = [0] + list(np.arange(0.25, 1, 0.25).round(2)) + list(range(1, 11))
egabas = [-74, -60, -46]
drugs_to_plot = [0.25, 4]


drugs_widget_part = partial(
    interactive,
    benzo_strengths=SelectMultiple(
        options=all_benzo_strengths,
        value=benzo_strengths,
        rows=len(all_benzo_strengths),
        description="Benzo strengths",
        style=style,
        layout=layout_width,
    ),
    egabas=SelectMultiple(
        options=egabas,
        value=egabas,
        rows=3,
        description="EGABA",
        style=style,
        layout=layout_width,
    ),
    picro_to_plot=FloatSlider(
        value=drugs_to_plot[0],
        min=0,
        max=1,
        step=0.25,
        description="picrotoxin",
        style=style,
    ),
    benzo_to_plot=IntSlider(
        value=drugs_to_plot[1],
        min=1,
        max=10,
        step=1,
        description="benzo",
        style=style,
    ),
)


def new_drugs_widget(f):
    drugs_widget = drugs_widget_part(f, manual_dict)

    drugs_widget.children[-2].button_style = manual_run_button_style

    # last 2 widgets (excluding Run and Output) are plotting params
    drugs_widget.children = (
        VBox(
            [
                HBox(
                    [
                        *drugs_widget.children[:2],
                        VBox(
                            [
                                HTML("<h3>Plotting Params</h3>"),
                                *drugs_widget.children[2:-2],
                            ],
                            layout=Layout(
                                border="1px solid gray",
                            ),
                        ),
                    ],
                    layout=Layout(
                        align_items="flex-start", justify_content="space-around"
                    ),
                ),
            ],
            # layout=Layout(align_items="flex-start"),
        ),
    ) + drugs_widget.children[-2:]

    return drugs_widget
