from functools import partial

from ipywidgets import (
    HTML,
    Checkbox,
    HBox,
    IntSlider,
    Layout,
    SelectionSlider,
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
# Tau KCC2 for different populations
############################


full_G_GABA_LIST = G_GABA_LIST + [37, 75, 150]

tau_widgets_part = partial(
    interactive,
    tau_KCC2_E_list=SelectMultiple(
        options=TAU_KCC2_LIST,
        value=TAU_KCC2_LIST,
        rows=len(TAU_KCC2_LIST),
        description="τKCC2 Exc",
    ),
    tau_KCC2_I_list=SelectMultiple(
        options=TAU_KCC2_LIST,
        value=TAU_KCC2_LIST,
        rows=len(TAU_KCC2_LIST),
        description="τKCC2 Inh",
    ),
    g_GABA_list=SelectMultiple(
        options=sorted(full_G_GABA_LIST),
        value=G_GABA_LIST,
        rows=len(full_G_GABA_LIST),
        description="gGABA",
    ),
    nrn_idx_i=SelectMultiple(
        options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        value=[0, 1, 2, 3],
        rows=5,
        description="Recorded Neurons",
        style=dict(description_width="initial"),
    ),
    default_tau_i=SelectionSlider(
        options=TAU_KCC2_LIST,
        value=60,
        description="Default τKCC2 I",
        style=style,
        layout=layout_auto_width,
    ),
    default_tau_e=SelectionSlider(
        options=TAU_KCC2_LIST,
        value=60,
        description="Default τKCC2 E",
        style=style,
        layout=layout_auto_width,
    ),
    default_ggaba=SelectionSlider(
        options=sorted(full_G_GABA_LIST),
        value=50,
        description="Default gGABA",
        style=style,
        layout=layout_auto_width,
    ),
    plot_ggaba=SelectMultiple(
        options=sorted(full_G_GABA_LIST),
        value=G_GABA_LIST,
        rows=len(full_G_GABA_LIST),
        description="plot gGABA",
    ),
    all_major_ticks=Checkbox(
        value=False,
        description="All τKCC2 as ticks",
    ),
    with_corner_traces=Checkbox(
        value=True,
        description="Full block (9) of traces (true) or only 5 (false), leaving out the corners",
        layout=layout_auto_width,
    ),
    run_idx=IntSlider(
        value=0,
        min=0,
        max=9,
        step=1,
        description="Example trace index (from 0)",
        style=style,
        layout=layout_auto_width,
    ),
    use_mean=Checkbox(
        value=False,
        description="Use mean (true) or default (false) for aggregate plots",
        layout=layout_auto_width,
    ),
    square_heatmap=Checkbox(
        value=False,
        description="Heatmap with square cells",
    ),
)


def new_tau_widget(f):
    tau_widgets = tau_widgets_part(f, manual_dict)

    # get the first 3 widgets as options to run and group them under the heading ("Params") in an HBox
    params_widgets = HBox(tau_widgets.children[:3], layout=Layout(margin="0 0 10px 0"))
    sim_params = HBox(tau_widgets.children[3:6])
    params_widgets = VBox(
        [HTML("<h3>Simulation Params</h3>"), params_widgets, sim_params],
        layout=Layout(border="1px solid gray"),
    )

    tau_widgets.children[-2].button_style = manual_run_button_style

    tau_widgets.children = (params_widgets,) + (
        VBox(
            [HTML("<h3>Plotting Params</h3>"), *tau_widgets.children[6:]],
        ),
    )

    return tau_widgets
