from functools import partial

from ipywidgets import (
    Button,
    Checkbox,
    HBox,
    IntSlider,
    Label,
    Layout,
    interactive,
)

from widgets.common import (
    layout_auto_width,
    manual_dict,
    manual_run_button_style,
    style_desc_width_auto,
)

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


explain_widget_part = partial(
    interactive,
    N=IntSlider(
        value=1000,
        min=100,
        max=10000,
        step=100,
        description="# neurons",
        layout=Layout(width="50%"),
    ),
    mv_step=IntSlider(
        value=initial_values["mv_step"], min=1, max=8, step=1, description="mV step"
    ),
    time_per_value=IntSlider(
        value=initial_values["time_per_value"],
        min=10,
        max=120,
        step=10,
        description="Time per value (ms)",
        style=style_desc_width_auto,
    ),
    egaba_start=IntSlider(
        value=initial_values["egaba_start"],
        min=-90,
        max=-30,
        step=1,
        description="EGABA start (inclusive)",
        style=style_desc_width_auto,
    ),
    egaba_end=IntSlider(
        value=initial_values["egaba_end"],
        min=-90,
        max=-30,
        step=1,
        description="EGABA end (exclusive)",
        style=style_desc_width_auto,
    ),
    plot_igaba=Checkbox(
        value=False,
        description="Plot I-GABA",
    ),
    plot_mean_rate=Checkbox(
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

    reset_button = Button(description="Figure 2", button_style="warning")
    reset_button.on_click(reset_values)
    supp_button = Button(description="Supplementary", button_style="info")
    supp_button.on_click(reset_values)

    num_neurons = explain_widget.children[0]
    explain_widget.children[-2].button_style = manual_run_button_style
    # add buttons and keep output last
    explain_widget.children = (
        explain_widget.children[1:-1]
        + (
            HBox(
                [
                    Label("Set above values to...", style={"font_weight": "bold"}),
                    reset_button,
                    supp_button,
                    Label("and change..."),
                    num_neurons,
                ],
                layout=Layout(border="1px solid gray", width="initial"),
            ),
        )
        + (explain_widget.children[-1],)
    )

    return explain_widget
