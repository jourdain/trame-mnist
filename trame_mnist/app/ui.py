from faulthandler import disable
from trame import controller as ctrl
from trame.layouts import SinglePage
from trame.html import vega, vuetify, xai, Div, Span

# Create single page layout type
layout = SinglePage("MNIST Exploration", on_ready=ctrl.on_ready)

# Toolbar
layout.title.set_text("MNIST Exploration")
with layout.toolbar as tb:
    tb.dense = True

    vuetify.VSpacer()

    # Training buttons
    with Div(v_show="view_mode === 'training'"):
        with vuetify.VBtn(
            small=True,
            outlined=True,
            icon=True,
            click="epoch_increase -= 1",
            disabled=("epoch_increase === 1",),
        ):
            vuetify.VIcon("mdi-minus")
        with vuetify.VBtn(
            small=True, outlined=True, icon=True, click="epoch_increase += 1"
        ):
            vuetify.VIcon("mdi-plus")
        with vuetify.VBtn(
            "Train ({{model_state.epoch}} +{{ epoch_increase }} epoch)",
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

    # Execution buttons
    with vuetify.VRow(
        v_show="view_mode === 'execution'",
        justify="end",
        align="center",
        classes="ma-0",
    ):
        with vuetify.VTooltip(bottom=True):
            with vuetify.Template(v_slot_activator="{ on, attrs }"):
                with Div(v_bind="attrs", v_on="on", __properties=["v_bind", "v_on"]):
                    vuetify.VCheckbox(
                        v_model=("prediction_search_failure", False),
                        classes="ml-4 my-0 py-0",
                        dense=True,
                        hide_details=True,
                        on_icon="mdi-target-account",
                        off_icon="mdi-target",
                    )
            Span("Toggle search for prediction mismatch")

        with vuetify.VTooltip(bottom=True):
            with vuetify.Template(v_slot_activator="{ on, attrs }"):
                with Div(v_bind="attrs", v_on="on", __properties=["v_bind", "v_on"]):
                    vuetify.VCheckbox(
                        v_model=("xai_viz", True),
                        classes="ml-4 my-0 py-0",
                        dense=True,
                        hide_details=True,
                        on_icon="mdi-wizard-hat",
                        off_icon="mdi-wizard-hat",
                    )
            Span("Toggle XAITK visualization")

    # Testing buttons
    with vuetify.VRow(
        v_show="view_mode === 'testing'",
        justify="end",
        align="center",
        classes="ma-0",
    ):
        vuetify.VChip(
            "{{ testing_count }}",
            classes="ma-2",
            color="green",
            text_color="white",
            hide_details=True,
            dense=True,
            click="testing_count = 0",
        )
        vuetify.VBtn(
            "Run testing",
            disabled=("testing_count > 0",),
            loading=("testing_running", False),
            click=ctrl.testing_run,
            hide_details=True,
            dense=True,
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
        with vuetify.VBtn(value="testing", disabled=("!prediction_available",)):
            vuetify.VIcon("mdi-ab-testing")


# Main content
with layout.content:
    with vuetify.VContainer(
        fluid=True,
        classes="pa-0",
    ):
        # Welcome for empty model
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

        # Training page
        with vuetify.VCol(v_if="view_mode == 'training' && model_state.epoch > 1"):
            chart_acc = vega.VegaEmbed(
                name="chart_acc",
                style="width: 100%;",
                v_show="view_mode === 'training'",
            )
            ctrl.chart_acc_update = chart_acc.update
            chart_loss = vega.VegaEmbed(
                name="chart_loss",
                style="width: 100%;",
                v_show="view_mode === 'training'",
            )
            ctrl.chart_loss_update = chart_loss.update

        # Execution page
        with vuetify.VRow(
            v_if="view_mode == 'execution'",
            classes="pa-3 ma-2",
            style="min-height: 275px;",
        ):
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
                    name="chart_prediction",
                    style="width: 100%; display: flex;",
                )
                ctrl.chart_pred_update = chart_pred.update

        # Execution page
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

        # Testing page
        with vuetify.VRow(
            v_if="view_mode == 'testing'",
            classes="pa-3 ma-2 overflow-hidden",
            align="center",
            justify="center",
            style="height: 100%; max-height: calc(100vh - 64px);",
        ):
            chart_confusion_matrix = vega.VegaEmbed(
                name="chart_confusion_matrix",
                v_show="view_mode == 'testing' && testing_count",
                style="width: 50%;",
                key="`${testing_count}-matrix`",
            )
            ctrl.chart_confusion_matrix = chart_confusion_matrix.update

            chart_class_accuracy = vega.VegaEmbed(
                name="chart_class_accuracy",
                v_show="view_mode == 'testing' && testing_count",
                style="width: 50%;",
                key="`${testing_count}-class`",
            )
            ctrl.chart_class_accuracy = chart_class_accuracy.update

# Footer
# layout.footer.hide()
