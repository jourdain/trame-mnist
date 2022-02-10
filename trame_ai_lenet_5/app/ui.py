from typing import ValuesView
from trame import state, controller as ctrl
from trame.layouts import SinglePage
from trame.html import vega, vuetify, Div, Span

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

    with Div(v_show="view_mode === 'execution'"):
        with vuetify.VBtn(
            icon=True,
            small=True,
            classes="ml-4",
            click=ctrl.prediction_next_failure,
            disabled=("!prediction_available",),
        ):
            vuetify.VIcon("mdi-shield-bug-outline")

        with vuetify.VBtn(
            small=True,
            classes="ml-4",
            click=ctrl.xai_run,
            disabled=("!prediction_available",),
            icon=True,
        ):
            vuetify.VIcon("mdi-wizard-hat")

    vuetify.VDivider(vertical=True, classes="mx-4")

    with vuetify.VBtnToggle(
        v_model=("view_mode", "training"),
        hide_details=True,
        dense=True,
        mandatory=True,
    ):
        with vuetify.VBtn(value="training"):
            vuetify.VIcon("mdi-school-outline")
        with vuetify.VBtn(value="execution", disabled=("!prediction_available",)):
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
                        with vuetify.VBtn(
                            fab=True,
                            click=ctrl.prediction_update,
                            style="position: absolute; right: -16px; top: -16px; z-index: 1;",
                            # color="primary",
                            x_small=True,
                            color=("prediction_success ? 'green' : 'red'",),
                            disabled=("!prediction_available",),
                        ):
                            vuetify.VIcon("mdi-autorenew")

                        with vuetify.VTooltip(bottom=True):
                            with vuetify.Template(v_slot_activator="{ on, attrs }"):
                                vuetify.VImg(
                                    v_bind="attrs",
                                    v_on="on",
                                    src=("prediction_input_url", ""),
                                    max_width=200,
                                    min_width=200,
                                    __properties=["v_bind", "v_on"],
                                )
                            Span("{{ prediction_label }}")

            with vuetify.VCol(align_self="center", cols=8):
                chart_pred = vega.VegaEmbed(
                    name="chart_prediction", style="width: 100%; display: flex;"
                )
                ctrl.chart_pred_update = chart_pred.update

# Footer
# layout.footer.hide()
