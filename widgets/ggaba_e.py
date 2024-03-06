"""
gGABA vs E-GABA vs TauKCC2 widgets
"""

from functools import partial

import brian2.numpy_ as np
from ipywidgets import (
    HTML,
    Checkbox,
    Dropdown,
    HBox,
    IntSlider,
    Layout,
    SelectionSlider,
    SelectMultiple,
    VBox,
    interactive,
)

from core.var_ranges import TAU_KCC2_LIST
from widgets.common import (
    has_description,
    layout_auto_width,
    manual_dict,
    manual_run_button_style,
    style_desc_width_auto,
)

##############################
# Default values
##############################
tau_KCC2_list = list(TAU_KCC2_LIST)

# add some more lower values for tau
ratio = tau_KCC2_list[1] / tau_KCC2_list[0]
# above ratio slightly off but results already cached.
# ratio = np.sqrt(2) # TODO: uncomment for final version
tau_KCC2_list = [np.round(tau_KCC2_list[0] / ratio, 1)] + tau_KCC2_list
tau_KCC2_list = [np.round(tau_KCC2_list[0] / ratio, 1)] + tau_KCC2_list
tau_KCC2_list = [np.round(tau_KCC2_list[0] / ratio, 1)] + tau_KCC2_list
tau_KCC2_list = [np.round(tau_KCC2_list[0] / ratio, 1)] + tau_KCC2_list

seeds = (None, 1234, 5678, 1426987, 86751, 16928, 98766, 876125, 127658, 9876)

g_GABA_options_dynamic = np.geomspace(10, 1000, 11).round(0)
g_GABA_options_static = sorted(
    set(
        np.append(
            np.round(np.arange(0, 100.0001, 10), 0),
            g_GABA_options_dynamic,
        )
    )
)

default_values = {
    "tau_KCC2s": tau_KCC2_list,
    "gGABAs": g_GABA_options_dynamic,
    "seeds": seeds,
    "gGABAsvEGABA": g_GABA_options_static,
    "time_per_value": 60,
    "mv_step": 2,
    "egaba_start": -80,
    "egaba_end": -38,
}

##############################
# Widgets definitions
##############################
gGABA_tauKCC2_widget_part = partial(
    interactive,
    gGABAsvEGABA=SelectMultiple(
        options=sorted(g_GABA_options_static),
        value=sorted(default_values["gGABAsvEGABA"]),
        rows=len(g_GABA_options_dynamic),
        description="scl - gGABAmax for static chloride",
        style=style_desc_width_auto,
    ),
    mv_step=IntSlider(
        value=default_values["mv_step"], min=1, max=8, step=1, description="scl - mV step"
    ),
    time_per_value=IntSlider(
        value=default_values["time_per_value"],
        min=10,
        max=120,
        step=10,
        description="scl - Time per value (ms)",
        style=style_desc_width_auto,
    ),
    egaba_start=IntSlider(
        value=default_values["egaba_start"],
        min=-90,
        max=-30,
        step=1,
        description="scl - EGABA start (inclusive)",
        style=style_desc_width_auto,
    ),
    egaba_end=IntSlider(
        value=default_values["egaba_end"],
        min=-90,
        max=-30,
        step=1,
        description="scl - EGABA end (exclusive)",
        style=style_desc_width_auto,
    ),
    gGABAs=SelectMultiple(
        options=sorted(g_GABA_options_dynamic),
        value=sorted(default_values["gGABAs"]),
        rows=len(g_GABA_options_dynamic),
        description="dcl - gGABAmax for dynamic cloride",
        style=style_desc_width_auto,
    ),
    tau_KCC2s=SelectMultiple(
        options=tau_KCC2_list,
        value=default_values["tau_KCC2s"],
        rows=len(tau_KCC2_list),
        description="dcl - τKCC2",
    ),
    seeds=SelectMultiple(
        options=seeds,
        value=default_values["seeds"],
        rows=5,
        description="Seeds",
    ),
    i_metric=Dropdown(
        options=[
            ("Mean", "mean"),
            ("Sum", "sum"),
            ("None (empty)", "diagram"),
        ],
        value="mean",
        description="plot - Pre-burst I GABA metric versus Number of bursts",
        style=style_desc_width_auto,
        layout=layout_auto_width,
    ),
    # plot tau_kcc2
    num_bursts=Dropdown(
        options=[("Mean", "mean"), ("Max", "max")],
        value="mean",
        description="plot - How to consider the number of bursts for gGABA vs τKCC2",
        style=style_desc_width_auto,
        layout=layout_auto_width,
    ),
    min_s=SelectionSlider(
        options=[("default", 0)] + [(f"{i}", i) for i in range(0, 22, 2)],
        value=0,
        description="plot - Scatter point size",
        style=style_desc_width_auto,
        layout=layout_auto_width,
    ),
    plot_3d=Checkbox(
        value=False,
        description="plot - Plot 3D",
        indent=False,
    ),
    bursts_max=SelectionSlider(
        options=[("default", 0)]
        + [(f"{np.power(2, i)}", np.power(2, i)) for i in range(0, 5)],
        value=0,
        description="plot - Upper value for color of 'number of bursts' in scatter plot",
        style=style_desc_width_auto,
        layout=layout_auto_width,
    ),
    egabas_plot=SelectionSlider(
        options=[(f"{i}", i) for i in range(1, 10)]
        + [("all", 0), ("paper", [-72, -66, -60, -56, -54, -52, -48, -44])],
        value=[-72, -66, -60, -56, -54, -52, -48, -44],
        description="plot - EGABAs to plot",
        style=style_desc_width_auto,
    ),
)


##############################
# Create and customize
##############################
def new_gGABA_tauKCC2_widget(f):
    gGABA_tauKCC2_widget = gGABA_tauKCC2_widget_part(f, manual_dict)

    gGABA_tauKCC2_widget.children[-2].button_style = manual_run_button_style

    # group static chloride params (scl - ) and remove the 'scl - ' from the description
    scl_widgets = []
    dcl_widgets = []
    plot_widgets = []
    other_widgets = []
    for widget in gGABA_tauKCC2_widget.children:
        if has_description(widget):
            if "scl" in widget.description:
                widget.description = widget.description.replace("scl - ", "")
                scl_widgets.append(widget)
            elif "dcl" in widget.description:
                widget.description = widget.description.replace("dcl - ", "")
                dcl_widgets.append(widget)
            elif "plot" in widget.description:
                widget.description = widget.description.replace("plot - ", "")
                plot_widgets.append(widget)
            else:
                other_widgets.append(widget)
        else:
            other_widgets.append(widget)

    gGABA_tauKCC2_widget.children = (
        HBox(
            [
                HTML("<h3>Static Cloride Params</h3>"),
                scl_widgets[0],
                VBox(scl_widgets[1:]),
            ],
            layout=Layout(border="1px solid gray"),
        ),
        HBox(
            [HTML("<h3>Dynamic Cloride Params</h3>"), *dcl_widgets],
            layout=Layout(border="1px solid gray"),
        ),
        HBox(
            [
                HTML("<h3>Plotting Params</h3>"),
                VBox(plot_widgets),
            ],
            layout=Layout(border="1px solid gray"),
        ),
        VBox([HTML("<h3>Other</h3>"), *other_widgets]),
    )

    return gGABA_tauKCC2_widget
