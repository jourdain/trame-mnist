from trame import state, controller as ctrl
from trame.layouts import SinglePage
from trame.html import vuetify, Div


# Create single page layout type
# (FullScreenPage, SinglePage, SinglePageWithDrawer)
layout = SinglePage("AI LeNet-5")

# Toolbar
layout.title.set_text("AI LeNet-5")
with layout.toolbar as tb:
    vuetify.VSpacer()
    vuetify.VSlider(v_model=("slider_value", 0), dense=True, hide_details=True)
    vuetify.VProgressLinear(
        value=("100 * model_state.epoch / epoch_end",),
        absolute=True,
        bottom=True,
        striped=True,
        active=("training_running",),
    )
    tb.add_child("{{ model_state.epoch }}/{{ epoch_end }}")
    vuetify.VBtn(
        "Train (+10 epoch)",
        loading=("training_running",),
        disabled=("training_running",),
        classes="ml-4",
        click=ctrl.start_training,
    )
    vuetify.VBtn(
        "Reset",
        classes="ml-4",
        click=ctrl.reset_training,
    )


# Main content
with layout.content:
    with vuetify.VContainer(fluid=True, classes="pa-0 fill-height"):
        Div(
            "Accuracy {{(model_state.training_accuracy[model_state.training_accuracy.length - 1])?.toFixed(2)}}%"
        )
        vuetify.VSparkline(
            value=("model_state.training_accuracy",),
            fill=True,
            line_width=1,
        )
        Div(
            "Loss  {{(model_state.training_loss[model_state.training_loss.length - 1])?.toFixed(2)}} - {{ model_state.training_loss.length }}"
        )
        vuetify.VSparkline(
            value=("model_state.training_loss",),
            fill=True,
            line_width=1,
        )

# Footer
# layout.footer.hide()
