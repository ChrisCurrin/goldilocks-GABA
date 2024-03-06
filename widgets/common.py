import ipywidgets

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
