import functools

import brian2.numpy_ as np
import ipywidgets

from widgets.common import (
    has_description,
    manual_dict,
    manual_run_button_style,
    style_desc_width_auto,
)

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
    E_leak=ipywidgets.IntSlider(
        -70, min=-80, max=-40, step=1, style=style_desc_width_auto
    ),
    g_l_I=ipywidgets.FloatSlider(
        20, min=0.2, max=30, step=0.2, style=style_desc_width_auto
    ),
    g_l_E=ipywidgets.FloatSlider(
        20, min=0.2, max=30, step=0.2, style=style_desc_width_auto
    ),  # leak params
    g_AMPA_max_ext=ipywidgets.FloatSlider(
        2,
        min=0,
        max=5,
        step=0.25,
        description="EXTERNAL Max conductance",
        style=style_desc_width_auto,
    ),
    C_ext=ipywidgets.IntSlider(800, 0, 1000, 10, style=style_desc_width_auto),
    rate_ext=(0, 5, 0.25),  # external input params
    manual_cl=[False, "both", "PC", "IN"],  # dynamic/manual arg
    E_Cl_0=(-100, -40, 1),
    E_Cl_end=(-100, -40, 1),
    num_ecl_steps=(1, 20, 1),  # dynamic + manual params
    dyn_cl=True,
    E_Cl_target=(-100, -40, 1),
    E_Cl_pop=["both", "PC", "IN"],  # ECl params
    length=ipywidgets.FloatSlider(
        value=7.5, min=0.5, max=30, step=0.5
    ),  # neuron params
    tau_KCC2_E=(5, 240, 5),
    tau_KCC2_I=(5, 240, 5),
    nrn_idx_i=ipywidgets.SelectMultiple(
        options=list(range(10)), value=[0, 1], style=style_desc_width_auto
    ),
    run_seed=(0, 100000, 1000),
)


def new_playground_widget(f):
    pg_widget = playground_widget_part(f, manual_dict)

    pg_widget.children[-2].button_style = manual_run_button_style

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
        and c.description in {"N", "duration", "dt", "run_seed", "nrn_idx_i"}
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
    stp_params = [
        c
        for c in pg_widget.children
        if has_description(c) and c.description in {"U_0", "tau_d", "tau_f"}
    ]

    ecl_clamp_params = [
        c
        for c in pg_widget.children
        if has_description(c)
        and c.description
        in {
            "E_Cl_0",
            "E_Cl_end",
            "manual_cl",
            "num_ecl_steps",
        }
    ]
    ecl_dyn_params = [
        c
        for c in pg_widget.children
        if has_description(c)
        and c.description
        in {
            "dyn_cl",
            "E_Cl_pop",
            "E_Cl_target",
            "tau_KCC2_E",
            "tau_KCC2_I",
        }
    ]
    ext_input_params = [
        c
        for c in pg_widget.children
        if has_description(c)
        and (
            c.description in {"g_AMPA_max_ext", "C_ext", "rate_ext"}
            or "EXTERNAL" in c.description
        )
    ]
    leak_params = [
        c
        for c in pg_widget.children
        if has_description(c) and c.description in {"g_l_I", "g_l_E", "E_leak"}
    ]
    neuron_params = [
        c
        for c in pg_widget.children
        if has_description(c) and c.description in {"length"}
    ]
    length_w = neuron_params[0]

    def length_to_volume(L, factor=1):
        a = L / 2  # Radius
        return factor * 4.0 / 3.0 * np.pi * (L / 2) * (a**2)  # volume oblate spheroid

    volume_e = ipywidgets.HTML(
        # value=f" Exc volume<br>{length_to_volume(length_w.value):.1f} μm3",
        layout=label_kwargs,
    )
    volume_i = ipywidgets.HTML(
        # value=f" Inh volume<br>{length_to_volume(length_w.value):.1f} μm3",
        layout=label_kwargs,
    )
    ipywidgets.link(
        (length_w, "value"),
        (volume_e, "value"),
        (lambda L: f"volume (μm3)<br>Exc<br>{length_to_volume(L):.1f}", lambda v: v),
    )
    ipywidgets.link(
        (length_w, "value"),
        (volume_i, "value"),
        (lambda L: f"<br>Inh<br>{length_to_volume(L, factor=2/3):.1f}", lambda v: v),
    )
    neuron_params += [volume_e, volume_i]

    all_children = (
        p_children
        + g_children
        + sim_params
        + mag_params
        + benzo_params
        + stp_params
        + ecl_clamp_params
        + ecl_dyn_params
        + ext_input_params
        + leak_params
        + neuron_params
    )
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
        elif _c.description == "nrn_idx_i":
            _c.description = "Record indices"
        elif _c.description == "num_ecl_steps":
            _c.description = "Steps"
        elif _c.description == "E_Cl_pop":
            _c.description = "Population"
        elif _c.description == "E_Cl_target":
            _c.description = "Target"
        elif _c.description == "dyn_cl":
            _c.description = "Dynamic?"
        elif _c.description == "E_Cl_0":
            _c.description = "Start"
        elif _c.description == "E_Cl_end":
            _c.description = "End"
        elif _c.description == "manual_cl":
            _c.description = "Manual?"
        elif _c.description in {"U_0", "tau_d", "tau_f"}:
            _c.description = (
                _c.description.replace("U_0", "Utilisation start")
                .replace("tau_d", "τ depression (s)")
                .replace("tau_f", "τ facilitation (s)")
            )
            _c.style.description_width = "initial"
        elif (
            _c.description in {"C_ext", "rate_ext", "g_AMPA_max_ext"}
            or "EXTERNAL" in _c.description
        ):
            if _c.description == "C_ext":
                _c.description = "# connections"
            elif _c.description == "rate_ext":
                _c.description = "Rate (Hz)"
            elif _c.description == "g_AMPA_max_ext":
                _c.description = "Max conductance (nS)"
            _c.description = _c.description.replace("EXTERNAL ", " ")
        elif _c.description in {"g_l_I", "g_l_E"}:
            _c.description = _c.description.replace("g_l_", "conductance ")
        elif _c.description == "E_leak":
            _c.description = "Reversal"
        elif _c.description.startswith("p_") or _c.description.startswith("w_"):
            _c.description = (
                _c.description.replace("p_", "")
                .replace("w_", "")
                .replace("ee", "PC→PC")
                .replace("ii", "IN→IN")
                .replace("ei", "IN→PC")
                .replace("ie", "PC→IN")
            )
        elif "g_" in _c.description:
            _c.description = _c.description.replace("g_", "").replace("_max", "")
        _c.description = (
            _c.description.replace("_Cl", "Cl⁻")
            .replace("_", " ")
            .replace("tau", "τ")
            .replace(" I", " [IN]")
            .replace(" E", " [PC]")
        )
    # link manual_cl and dyn_cl widgets
    ipywidgets.link(
        (ecl_clamp_params[0], "value"),
        (ecl_dyn_params[0], "value"),
        (
            lambda v: not v if isinstance(v, bool) else False,
            lambda v: False if v else "both",
        ),
    )

    pg_widget.children = (
        [
            ipywidgets.HBox(
                [
                    ipywidgets.HTML("<h1>Simulation parameters</h1>", **label_kwargs),
                    *sim_params,
                ],
                layout=hbox_layout,
            )
        ]
        + [
            ipywidgets.VBox(
                [
                    ipywidgets.HTML("<h1>Connections</h1>", **label_kwargs),
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
                            ipywidgets.Label("max conductance (nS)", **label_kwargs),
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
                    ipywidgets.HTML("<h1>0 mM Mg2⁺ wash</h1>", **label_kwargs),
                    *mag_params,
                ],
                layout=hbox_layout,
            )
        ]
        + [
            ipywidgets.HBox(
                [
                    ipywidgets.HTMLMath(
                        value="<h1>Benzo</h1><br><em>modify max GABA conductance</em>",
                        **label_kwargs,
                    ),
                    *benzo_params,
                ],
                layout=hbox_layout,
            )
        ]
        + [
            ipywidgets.HBox(
                [
                    ipywidgets.HTML("<h1>STP</h1>", **label_kwargs),
                    *stp_params,
                ],
                layout=hbox_layout,
            )
        ]
        + [
            ipywidgets.VBox(
                [
                    ipywidgets.HTMLMath("<h1>ECl⁻", **label_kwargs),
                    ipywidgets.HBox(
                        [
                            ipywidgets.Label("Static", **label_kwargs),
                            *ecl_clamp_params,
                        ]
                    ),
                    ipywidgets.HBox(
                        [
                            ipywidgets.Label("Dynamic", **label_kwargs),
                            *ecl_dyn_params,
                        ]
                    ),
                ],
                layout=hbox_layout,
            )
        ]
        + [
            ipywidgets.HBox(
                [
                    ipywidgets.HTML("<h1>External Input</h1>", **label_kwargs),
                    *ext_input_params,
                ],
                layout=hbox_layout,
            )
        ]
        + [
            ipywidgets.HBox(
                [
                    ipywidgets.HTML("<h1>Leak</h1>", **label_kwargs),
                    *leak_params,
                ],
                layout=hbox_layout,
            )
        ]
        + [
            ipywidgets.HBox(
                [
                    ipywidgets.HTML("<h1>Dimensions</h1>", **label_kwargs),
                    *neuron_params,
                ],
                layout=hbox_layout,
            )
        ]
        + [c for c in pg_widget.children if c not in all_children_set]
    )
    return pg_widget
