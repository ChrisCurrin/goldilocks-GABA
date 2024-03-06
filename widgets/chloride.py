from functools import partial

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

from core.var_ranges import G_GABA_LIST, TAU_KCC2_LIST
from widgets.common import (
    layout_auto_width,
    manual_dict,
    manual_run_button_style,
    style,
)

############################
# Dynamic Chloride
############################

tau_KCC2s = list(TAU_KCC2_LIST)[::2]
E_Cl_0s = (-60, -74, -88)
E_Cl_0s_selected = (-60, -88)
g_GABAs = (50, 25, 100)
seeds = (None, 1038, 1337, 1111, 1010, 1011, 1101, 1110, 11110, 111100)
burst_window = 60
duration = 600

# plotting preferences
default_tau = 60
stripplot_alpha = 0.4
stripplot_size = 3
stripplot_jitter = 0.4

chloride_widget_part = partial(
    interactive,
    tau_KCC2s=SelectMultiple(
        options=TAU_KCC2_LIST, value=tau_KCC2s, rows=5, description="tau KCC2"
    ),
    E_Cl_0s=SelectMultiple(
        options=E_Cl_0s, value=E_Cl_0s_selected, rows=3, description="E Cl-"
    ),
    g_GABAs=SelectMultiple(
        options=G_GABA_LIST, value=g_GABAs, rows=3, description="G GABA"
    ),
    seeds=SelectMultiple(
        options=seeds,
        value=seeds,
        rows=5,
        description="Seeds",
        layout=layout_auto_width,
    ),
    duration=IntSlider(
        value=duration,
        min=100,
        max=1000,
        step=100,
        description="Duration (ms)",
        style=style,
        # layout=layout,
    ),
    burst_window=IntSlider(
        value=burst_window,
        min=10,
        max=120,
        step=10,
        description="Burst window (ms)",
        style=style,
        layout=layout_auto_width,
    ),
    default_tau=IntSlider(
        value=default_tau,
        min=10,
        max=120,
        step=10,
        description="Default tau (ms)",
        style=style,
        layout=layout_auto_width,
    ),
    stripplot_alpha=FloatSlider(
        value=stripplot_alpha,
        min=0,
        max=1,
        step=0.1,
        description="Stripplot alpha",
        style=style,
        layout=layout_auto_width,
    ),
    stripplot_size=FloatSlider(
        value=stripplot_size,
        min=1,
        max=10,
        step=1,
        description="Stripplot size",
        style=style,
        layout=layout_auto_width,
    ),
    stripplot_jitter=FloatSlider(
        value=stripplot_jitter,
        min=0,
        max=1,
        step=0.1,
        description="Stripplot jitter",
        style=style,
        layout=layout_auto_width,
    ),
)


def new_chloride_widget(f):
    chloride_widget = chloride_widget_part(f, manual_dict)

    # Run button color
    chloride_widget.children[-2].button_style = manual_run_button_style

    # everything before "Duration" param
    params = VBox(
        [
            HTML("<h3>Simulation Params</h3>"),
            HBox(chloride_widget.children[:3]),
            HBox(chloride_widget.children[3:5]),
        ],
    )
    # everything after "Duration" param
    plotting_params = VBox(
        [
            HTML("<h3>Plotting Params</h3>"),
            *chloride_widget.children[5:-2],
        ],
        layout=Layout(border="1px solid gray"),
    )

    chloride_widget.children = (
        VBox(
            [
                VBox(
                    [params, plotting_params],
                ),
                *chloride_widget.children[-2:],
            ]
        ),
    )

    return chloride_widget
