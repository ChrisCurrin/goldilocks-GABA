import functools
import warnings

import brian2.numpy_ as np
import ipywidgets
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

from settings import COLOR, G_GABA_LIST, TAU_KCC2_LIST, logging, text

############################
# Global defaults
############################

manual_dict = dict(manual=True, manual_name="Run")
manual_run_button_style = "primary"
style = dict(description_width="120px")
style_desc_width_auto = dict(description_width="auto")
layout_auto_width = ipywidgets.Layout(width="auto")
layout_width = ipywidgets.Layout(width="200px")


def has_description(x):
    return isinstance(x, ipywidgets.widget_description.DescriptionWidget)


############################
# Explain Figure Widgets
############################

initial_values = {
    "mv_step": 4,
    "time_per_value": 40,
    "egaba_start": -78,
    "egaba_end": -34,
}
supp_values = {
    "mv_step": 2,
    "time_per_value": 60,
    "egaba_start": -74,
    "egaba_end": -40,
}


explain_widget_part = functools.partial(
    ipywidgets.interactive,
    N=ipywidgets.IntSlider(
        value=1000,
        min=100,
        max=10000,
        step=100,
        description="# neurons",
        layout=ipywidgets.Layout(width="50%"),
    ),
    mv_step=ipywidgets.IntSlider(
        value=initial_values["mv_step"], min=1, max=8, step=1, description="mV step"
    ),
    time_per_value=ipywidgets.IntSlider(
        value=initial_values["time_per_value"],
        min=10,
        max=120,
        step=10,
        description="Time per value (ms)",
        style=style_desc_width_auto,
    ),
    egaba_start=ipywidgets.IntSlider(
        value=initial_values["egaba_start"],
        min=-90,
        max=-30,
        step=1,
        description="EGABA start (inclusive)",
        style=style_desc_width_auto,
    ),
    egaba_end=ipywidgets.IntSlider(
        value=initial_values["egaba_end"],
        min=-90,
        max=-30,
        step=1,
        description="EGABA end (exclusive)",
        style=style_desc_width_auto,
    ),
    plot_igaba=ipywidgets.Checkbox(
        value=False,
        description="Plot I-GABA",
    ),
    plot_mean_rate=ipywidgets.Checkbox(
        value=True,
        description="Plot mean firing rate rate of population",
        layout=layout_auto_width,
    ),
)


def new_explain_widget(f):
    explain_widget = explain_widget_part(f, manual_dict)

    def reset_values(_button):
        """Reset the interactive plots to inital values."""
        value_dict = supp_values if "Supp" in _button.description else initial_values
        for widget, value in zip(explain_widget.children, value_dict.values()):
            widget.value = value

    reset_button = ipywidgets.Button(description="Figure 2", button_style="warning")
    reset_button.on_click(reset_values)
    supp_button = ipywidgets.Button(description="Supplementary", button_style="info")
    supp_button.on_click(reset_values)

    num_neurons = explain_widget.children[0]
    explain_widget.children[-2].button_style = manual_run_button_style
    # add buttons and keep output last
    explain_widget.children = (
        explain_widget.children[1:-1]
        + (
            ipywidgets.HBox(
                [
                    ipywidgets.Label(
                        "Set above values to...", style={"font_weight": "bold"}
                    ),
                    reset_button,
                    supp_button,
                    ipywidgets.Label("and change..."),
                    num_neurons,
                ],
                layout=ipywidgets.Layout(border="1px solid gray", width="initial"),
            ),
        )
        + (explain_widget.children[-1],)
    )

    return explain_widget


############################
# Drugs Figure Widgets
############################

benzo_strengths = (0, 0.25, 0.5, 1, 2, 4, 8)
all_benzo_strengths = [0] + list(np.arange(0.25, 1, 0.25).round(2)) + list(range(1, 11))
egabas = [-74, -60, -46]
drugs_to_plot = [0.25, 4]


drugs_widget_part = functools.partial(
    ipywidgets.interactive,
    benzo_strengths=ipywidgets.SelectMultiple(
        options=all_benzo_strengths,
        value=benzo_strengths,
        rows=len(all_benzo_strengths),
        description="Benzo strengths",
        style=style,
        layout=layout_width,
    ),
    egabas=ipywidgets.SelectMultiple(
        options=egabas,
        value=egabas,
        rows=3,
        description="EGABA",
        style=style,
        layout=layout_width,
    ),
    picro_to_plot=ipywidgets.FloatSlider(
        value=drugs_to_plot[0],
        min=0,
        max=1,
        step=0.25,
        description="picrotoxin",
        style=style,
    ),
    benzo_to_plot=ipywidgets.IntSlider(
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
        ipywidgets.VBox(
            [
                ipywidgets.HBox(
                    [
                        *drugs_widget.children[:2],
                        ipywidgets.VBox(
                            [
                                ipywidgets.HTML("<h3>Plotting Params</h3>"),
                                *drugs_widget.children[2:-2],
                            ],
                            layout=ipywidgets.Layout(
                                border="1px solid gray",
                            ),
                        ),
                    ],
                    layout=ipywidgets.Layout(
                        align_items="flex-start", justify_content="space-around"
                    ),
                ),
            ],
            # layout=ipywidgets.Layout(align_items="flex-start"),
        ),
    ) + drugs_widget.children[-2:]

    return drugs_widget


############################
# Dynamic Chloride
############################

tau_KCC2s = TAU_KCC2_LIST[::2]
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

chloride_widget_part = functools.partial(
    ipywidgets.interactive,
    tau_KCC2s=ipywidgets.SelectMultiple(
        options=TAU_KCC2_LIST, value=tau_KCC2s, rows=5, description="tau KCC2"
    ),
    E_Cl_0s=ipywidgets.SelectMultiple(
        options=E_Cl_0s, value=E_Cl_0s_selected, rows=3, description="E Cl-"
    ),
    g_GABAs=ipywidgets.SelectMultiple(
        options=G_GABA_LIST, value=g_GABAs, rows=3, description="G GABA"
    ),
    seeds=ipywidgets.SelectMultiple(
        options=seeds,
        value=seeds,
        rows=5,
        description="Seeds",
        layout=layout_auto_width,
    ),
    duration=ipywidgets.IntSlider(
        value=duration,
        min=100,
        max=1000,
        step=100,
        description="Duration (ms)",
        style=style,
        # layout=layout,
    ),
    burst_window=ipywidgets.IntSlider(
        value=burst_window,
        min=10,
        max=120,
        step=10,
        description="Burst window (ms)",
        style=style,
        layout=layout_auto_width,
    ),
    default_tau=ipywidgets.IntSlider(
        value=default_tau,
        min=10,
        max=120,
        step=10,
        description="Default tau (ms)",
        style=style,
        layout=layout_auto_width,
    ),
    stripplot_alpha=ipywidgets.FloatSlider(
        value=stripplot_alpha,
        min=0,
        max=1,
        step=0.1,
        description="Stripplot alpha",
        style=style,
        layout=layout_auto_width,
    ),
    stripplot_size=ipywidgets.FloatSlider(
        value=stripplot_size,
        min=1,
        max=10,
        step=1,
        description="Stripplot size",
        style=style,
        layout=layout_auto_width,
    ),
    stripplot_jitter=ipywidgets.FloatSlider(
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
    params = ipywidgets.VBox(
        [
            ipywidgets.HTML("<h3>Simulation Params</h3>"),
            ipywidgets.HBox(chloride_widget.children[:3]),
            ipywidgets.HBox(chloride_widget.children[3:5]),
        ],
    )
    # everything after "Duration" param
    plotting_params = ipywidgets.VBox(
        [
            ipywidgets.HTML("<h3>Plotting Params</h3>"),
            *chloride_widget.children[5:-2],
        ],
        layout=ipywidgets.Layout(border="1px solid gray"),
    )

    chloride_widget.children = (
        ipywidgets.VBox(
            [
                ipywidgets.VBox(
                    [params, plotting_params],
                ),
                *chloride_widget.children[-2:],
            ]
        ),
    )

    return chloride_widget


############################
# Tau KCC2 for different populations
############################


full_G_GABA_LIST = G_GABA_LIST + [37, 75, 150]

tau_widgets_part = functools.partial(
    ipywidgets.interactive,
    tau_KCC2_E_list=ipywidgets.SelectMultiple(
        options=TAU_KCC2_LIST,
        value=TAU_KCC2_LIST,
        rows=len(TAU_KCC2_LIST),
        description="τKCC2 Exc",
    ),
    tau_KCC2_I_list=ipywidgets.SelectMultiple(
        options=TAU_KCC2_LIST,
        value=TAU_KCC2_LIST,
        rows=len(TAU_KCC2_LIST),
        description="τKCC2 Inh",
    ),
    g_GABA_list=ipywidgets.SelectMultiple(
        options=sorted(full_G_GABA_LIST),
        value=G_GABA_LIST,
        rows=len(full_G_GABA_LIST),
        description="gGABA",
    ),
    nrn_idx_i=ipywidgets.SelectMultiple(
        options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        value=[0, 1, 2, 3],
        rows=5,
        description="Recorded Neurons",
        style=dict(description_width="initial"),
    ),
    default_tau_i=ipywidgets.SelectionSlider(
        options=TAU_KCC2_LIST,
        value=60,
        description="Default τKCC2 I",
        style=style,
        layout=layout_auto_width,
    ),
    default_tau_e=ipywidgets.SelectionSlider(
        options=TAU_KCC2_LIST,
        value=60,
        description="Default τKCC2 E",
        style=style,
        layout=layout_auto_width,
    ),
    default_ggaba=ipywidgets.SelectionSlider(
        options=sorted(full_G_GABA_LIST),
        value=50,
        description="Default gGABA",
        style=style,
        layout=layout_auto_width,
    ),
    plot_ggaba=ipywidgets.SelectMultiple(
        options=sorted(full_G_GABA_LIST),
        value=G_GABA_LIST,
        rows=len(full_G_GABA_LIST),
        description="plot gGABA",
    ),
    all_major_ticks=ipywidgets.Checkbox(
        value=False,
        description="All τKCC2 as ticks",
    ),
    with_corner_traces=ipywidgets.Checkbox(
        value=True,
        description="Full block (9) of traces (true) or only 5 (false), leaving out the corners",
        layout=layout_auto_width,
    ),
    run_idx=ipywidgets.IntSlider(
        value=0,
        min=0,
        max=9,
        step=1,
        description="Example trace index (from 0)",
        style=style,
        layout=layout_auto_width,
    ),
    use_mean=ipywidgets.Checkbox(
        value=False,
        description="Use mean (true) or default (false) for aggregate plots",
        layout=layout_auto_width,
    ),
    square_heatmap=ipywidgets.Checkbox(
        value=False,
        description="Heatmap with square cells",
    ),
)


def new_tau_widget(f):
    tau_widgets = tau_widgets_part(f, manual_dict)

    # get the first 3 widgets as options to run and group them under the heading ("Params") in an HBox
    params_widgets = ipywidgets.HBox(
        tau_widgets.children[:3], layout=ipywidgets.Layout(margin="0 0 10px 0")
    )
    sim_params = ipywidgets.HBox(tau_widgets.children[3:6])
    params_widgets = ipywidgets.VBox(
        [ipywidgets.HTML("<h3>Simulation Params</h3>"), params_widgets, sim_params],
        layout=ipywidgets.Layout(border="1px solid gray"),
    )

    tau_widgets.children[-2].button_style = manual_run_button_style

    tau_widgets.children = (params_widgets,) + (
        ipywidgets.VBox(
            [ipywidgets.HTML("<h3>Plotting Params</h3>"), *tau_widgets.children[6:]],
        ),
    )

    return tau_widgets


############################
# Playground widget
############################


playground_widget_part = functools.partial(
    ipywidgets.interactive,
    N=(500, 10000, 500),
    duration=(5, 600, 5),
    dt=(0.001, 0.02, 0.001),  # Simulation params
    Mg2_t0=(0, 5, 1),
    zero_mag_wash_rate=(1, 10, 1),  # Seizure params
    benzo_onset_t=(0, 100, 5),
    benzo_wash_rate=(1, 10, 1),
    benzo_strength=(0, 10, 0.25),
    benzo_off_t=(0, 600, 5),  # drug params
    # p=(0.01, 0.1, 0.01),
    p_ee=(0.01, 0.1, 0.01),
    p_ei=(0.01, 0.1, 0.01),
    p_ie=(0.01, 0.1, 0.01),
    p_ii=(0.01, 0.1, 0.01),  # Connection params
    # w=(0, 5, 0.5),
    w_ee=(0, 5, 0.5),
    w_ei=(0, 5, 0.5),
    w_ie=(0, 5, 0.5),
    w_ii=(0, 5, 0.5),  # weight params
    U_0=(0, 1, 0.01),
    tau_d=(0, 20, 0.5),
    tau_f=(0, 5, 0.5),  # STP params
    g_AMPA_max=(0, 50, 1),
    g_NMDA_max=(0, 10, 1),
    g_GABA_max=(0, 200, 25),  # conductances
    E_Cl_0=(-100, -40, 1),
    E_Cl_target=(-100, -40, 1),
    E_Cl_end=(-100, -40, 1),
    E_Cl_pop=["None", "both", "E", "I"],  # ECl params
    length=(3, 15, 1),  # neuron params
    # dyn_cl=[True, False],
    manual_cl=[False, True, "both", "E", "I"],  # dynamic/manual arg
    tau_KCC2_E=(5, 240, 5),
    tau_KCC2_I=(5, 240, 5),
    num_ecl_steps=(1, 20, 1),  # dynamic + manual params
    nrn_idx_i=ipywidgets.SelectMultiple(
        options=list(range(10)), value=[0, 1], style=style_desc_width_auto
    ),
    run_seed=(0, 100000, 1000),
)


def new_playground_widget(f):
    pg_widget = playground_widget_part(f, manual_dict)

    hbox_layout = ipywidgets.Layout(
        width="auto",
        padding="0px",
        margin="0px",
        grid_gap="0px",
        border="1px solid black",
    )
    children_layout = ipywidgets.Layout(
        width="auto",
        height="auto",
        padding="0px",
        margin="0px",
        display="grid",
        grid_gap="0px",
        grid_template_columns="initial",
    )
    label_kwargs = dict(
        style={"font_weight": "bold"}, layout=ipywidgets.Layout(width="200px")
    )
    p_children = [
        c
        for c in pg_widget.children
        if has_description(c)
        and c.description
        in {"p_ee", "p_ei", "p_ie", "p_ii", "w_ee", "w_ei", "w_ie", "w_ii"}
    ]
    g_children = [
        c
        for c in pg_widget.children
        if has_description(c)
        and c.description in {"g_AMPA_max", "g_NMDA_max", "g_GABA_max"}
    ]

    sim_params = [
        c
        for c in pg_widget.children
        if has_description(c)
        and c.description in {"N", "duration", "dt", "nrn_idx_i", "run_seed"}
    ]
    mag_params = [
        c
        for c in pg_widget.children
        if has_description(c) and c.description in {"Mg2_t0", "zero_mag_wash_rate"}
    ]
    benzo_params = [
        c
        for c in pg_widget.children
        if has_description(c)
        and c.description
        in {"benzo_onset_t", "benzo_wash_rate", "benzo_strength", "benzo_off_t"}
    ]
    all_children = p_children + g_children + sim_params + mag_params + benzo_params
    all_children_set = set(all_children)
    for _c in all_children:
        _c.layout = children_layout

        if _c.description == "Mg2_t0":
            _c.description = "Start time"
        elif _c.description == "zero_mag_wash_rate":
            _c.description = "Wash rate"
        elif "benzo" in _c.description:
            _c.description = (
                _c.description.replace("_t", " time")
                .replace("_", " ")
                .replace("benzo", "")
                .title()
            )
        elif "g_" in _c.description:
            _c.description = _c.description.replace("g_", "").replace("_max", "")
        elif _c.description == "nrn_idx_i":
            _c.description = "Record indices"

    pg_widget.children = (
        [
            ipywidgets.HBox(
                [
                    ipywidgets.Label("Simulation parameters", **label_kwargs),
                    *sim_params,
                ],
                layout=hbox_layout,
            )
        ]
        + [
            ipywidgets.VBox(
                [
                    ipywidgets.Label("Connection", **label_kwargs),
                    ipywidgets.HBox(
                        [
                            ipywidgets.Label("probabilities", **label_kwargs),
                            *p_children[:4],
                        ],
                        # layout=hbox_layout,
                    ),
                    ipywidgets.HBox(
                        [
                            ipywidgets.Label("weights", **label_kwargs),
                            *p_children[4:],
                        ],
                        # layout=hbox_layout,
                    ),
                    ipywidgets.HBox(
                        [
                            ipywidgets.Label("max conductance", **label_kwargs),
                            *g_children,
                        ]
                    ),
                ],
                layout=hbox_layout,
            )
        ]
        + [
            ipywidgets.HBox(
                [
                    ipywidgets.Label("0 mM Mg2+ wash", **label_kwargs),
                    *mag_params,
                ],
                layout=hbox_layout,
            )
        ]
        + [
            ipywidgets.HBox(
                [ipywidgets.Label("Benzo", **label_kwargs), *benzo_params],
                layout=hbox_layout,
            )
        ]
        + [c for c in pg_widget.children if c not in all_children_set]
    )
    return pg_widget
