from typing import ValuesView
from trame import state, controller as ctrl
from trame.layouts import SinglePage
from trame.html import vega, vuetify, Div

# Create single page layout type
layout = SinglePage("AI LeNet-5", on_ready=ctrl.on_ready)

# Toolbar
layout.title.set_text("AI LeNet-5")
with layout.toolbar as tb:
    tb.dense = True

    vuetify.VSpacer()

    with Div(v_show="view_mode === 'training'"):
        with vuetify.VBtn(
            "Train ({{model_state.epoch}} +10 epoch)",
            loading=("training_running",),
            disabled=("training_running",),
            classes="ml-4",
            click=ctrl.training_add,
            style="overflow: hidden",
        ):
            with vuetify.Template(v_slot_loader=True):
                vuetify.VProgressLinear(
                    "{{ model_state.epoch }}/{{ epoch_end }}",
                    value=("100 * model_state.epoch / epoch_end",),
                    striped=True,
                    stream=True,
                    buffer_value=0,
                    height=36,
                )

        vuetify.VBtn(
            "Reset",
            classes="ml-4",
            disabled=("training_running",),
            click=ctrl.training_reset,
        )

    vuetify.VDivider(vertical=True, classes="mx-4")

    with vuetify.VBtnToggle(
        v_model=("view_mode", "training"),
        hide_details=True,
        dense=True,
        mandatory=True,
    ):
        with vuetify.VBtn(value="training"):
            vuetify.VIcon("mdi-school-outline")
        with vuetify.VBtn(value="execution"):
            vuetify.VIcon("mdi-run")


# Main content
with layout.content:
    with vuetify.VContainer(fluid=True, classes="pa-0"):
        with vuetify.VCol(v_if="view_mode == 'training'"):
            chart_acc = vega.VegaEmbed(name="chart_acc", style="width: 100%;")
            ctrl.chart_acc_update = chart_acc.update
            chart_loss = vega.VegaEmbed(name="chart_loss", style="width: 100%;")
            ctrl.chart_loss_update = chart_loss.update
        with vuetify.VRow(v_if="view_mode == 'execution'", classes="pa-3 ma-2 "):
            with vuetify.VCol(align_self="center", cols=4):
                with vuetify.VRow(justify="center"):
                    with Div(style="position: relative; flex: none; width: 200px;"):
                        vuetify.VImg(
                            src=("prediction_input_url", ""),
                            max_width=200,
                            min_width=200,
                        )
                        with vuetify.VBtn(
                            fab=True,
                            click=ctrl.prediction_update_input,
                            style="position: absolute; right: -16px; top: -16px; z-index: 1;",
                            # color="primary",
                            x_small=True,
                        ):
                            vuetify.VIcon("mdi-autorenew")

            with vuetify.VCol(align_self="center", cols=2):
                with vuetify.VRow(justify="center"):
                    with vuetify.VBtn(
                        icon=True,
                        click=ctrl.prediction_run,
                        outlined=True,
                    ):
                        vuetify.VIcon("mdi-magic-staff")

            with vuetify.VCol(align_self="center", cols=6):
                chart_pred = vega.VegaEmbed(
                    name="chart_prediction", style="width: 100%; display: flex;"
                )
                ctrl.chart_pred_update = chart_pred.update

# Footer
# layout.footer.hide()
