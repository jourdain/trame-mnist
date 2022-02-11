from typing import ValuesView
from trame import state, controller as ctrl
from trame.layouts import SinglePage
from trame.html import vega, vuetify, xai, Div, Span

# Create single page layout type
layout = SinglePage("MNIST Exploration", on_ready=ctrl.on_ready)

# Toolbar
layout.title.set_text("MNIST Exploration")
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

    with vuetify.VRow(
        v_show="view_mode === 'execution'", justify="end", align="center"
    ):
        with vuetify.VBtn(
            icon=True,
            small=True,
            classes="ml-4",
            click=ctrl.prediction_next_failure,
            disabled=("!prediction_available",),
        ):
            vuetify.VIcon("mdi-shield-bug-outline")

        vuetify.VCheckbox(
            v_model=("xai_viz", True),
            classes="ml-4 my-0 py-0",
            dense=True,
            hide_details=True,
            on_icon="mdi-wizard-hat",
            off_icon="mdi-snapchat",
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
        with vuetify.VBtn(value="execution", disabled=("!prediction_available",)):
            vuetify.VIcon("mdi-run")


# Main content
with layout.content:
    with vuetify.VContainer(fluid=True, classes="pa-0"):
        with vuetify.VCard(v_if=("model_state.epoch < 2",), classes="ma-8"):
            vuetify.VCardTitle("Getting started")
            vuetify.VCardText(
                """
                To get started, you need to create an AI model by trainning it.
                The "Train" button let you add more learning to it.
                But to start seeing learning progress and explore its prediction, you need to do it at least one.
            """
            )
            with vuetify.VCardActions():
                vuetify.VSpacer()
                vuetify.VBtn(
                    "Start training",
                    click=ctrl.training_add,
                    disabled=("training_running",),
                )

        with vuetify.VCol(v_if="view_mode == 'training' && model_state.epoch > 1"):
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

        with vuetify.VCol(v_if="view_mode == 'execution'"):
            with vuetify.VRow(classes="px-0 ma-0"):
                with vuetify.VCol(v_for=("i in 10",), key="i", classes="pa-0 ma-0"):
                    with vuetify.VRow(
                        justify="center", align="center", classes="pa-0 ma-0"
                    ):
                        Div(
                            "{{ i - 1 }}",
                            style="width: 70%; text-align: center;",
                            classes=(
                                "{ 'rounded-pill': true, 'elevation-5 teal': prediction_label == (i - 1) }",
                            ),
                        )
            with vuetify.VCol(v_if=("xai_viz",), classes="px-2 ma-0"):
                with vuetify.VTooltip(
                    v_for=("result, method, idx in xai_results",),
                    top=("idx === 0",),
                    bottom=("idx === 1",),
                    key="method",
                ):
                    with vuetify.Template(v_slot_activator="{ on, attrs }"):
                        with vuetify.VRow(
                            align_self="center",
                            v_bind=("attrs",),
                            v_on=("on",),
                            __properties=["v_bind", "v_on"],
                            classes="py-2",
                        ):
                            with vuetify.VCol(
                                v_for=("i in 10",), key="i", classes="py-0"
                            ):
                                xai.XaiImage(
                                    src=("prediction_input_url",),
                                    areas=("[]",),
                                    width="100%",
                                    heatmaps=("result.heatmaps",),
                                    heatmap_opacity=0.85,
                                    heatmap_color_preset="BuRd",  # coolwarm, rainbow, blue2cyan, BuRd
                                    heatmap_active=("`${i-1}`",),
                                    heatmap_color_range=("[-1, 1]",),
                                    heatmap_color_mode="custom",
                                )
                    Span(
                        "{{ method }}: {{ result.range }} of range but colored [-1, 1]"
                    )


# Footer
# layout.footer.hide()
