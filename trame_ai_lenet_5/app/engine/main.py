import os
import asyncio
from pathlib import Path

import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor

from . import ml, utils

import altair as alt
from trame import state, controller as ctrl

MODEL_PATH = str(Path(ml.DATA_DIR, "model_lenet-5.trained").resolve().absolute())
PENDING_TASKS = []

# -----------------------------------------------------------------------------
# Initial state
# -----------------------------------------------------------------------------
state.update(
    {
        "epoch_end": 0,
        "training_running": False,
        "model_state": {
            "epoch": 0,
            "training_accuracy": [],
            "training_loss": [],
            "validation_accuracy": [],
            "validation_loss": [],
        },
    }
)

# -----------------------------------------------------------------------------
# Add epoch to training
# -----------------------------------------------------------------------------
async def run_training():
    await asyncio.gather(*PENDING_TASKS)
    PENDING_TASKS.clear()

    if state.model_state.get("epoch") >= state.epoch_end:
        state.epoch_end += 10

    m = multiprocessing.Manager()
    queue = m.Queue()
    loop = asyncio.get_event_loop()
    monitor = loop.create_task(utils.monitor_state_queue(queue))
    training = loop.run_in_executor(
        ProcessPoolExecutor(1),
        partial(ml.train_model, MODEL_PATH, queue, state.epoch_end),
    )

    PENDING_TASKS.append(monitor)
    PENDING_TASKS.append(training)


# -----------------------------------------------------------------------------
# Remove saved model and reset local state
# -----------------------------------------------------------------------------
def reset_training():
    if Path(MODEL_PATH).exists():
        os.remove(MODEL_PATH)

    state.epoch_end = 0
    state.model_state = {
        "epoch": 0,
        "training_accuracy": [],
        "training_loss": [],
        "validation_accuracy": [],
        "validation_loss": [],
    }


@state.change("slider_value")
def monitor_slider(slider_value, **kwargs):
    print(slider_value)


@state.change("model_state")
def update_chart(model_state, **kwargs):
    source = []
    for serie in [
        "training_accuracy",
        "training_loss",
        "validation_accuracy",
        "validation_loss",
    ]:
        values = model_state.get(serie)
        for epoch, value in enumerate(values):
            source.append(
                {
                    "serie": serie.replace("_", " "),
                    "epoch": epoch + 1,
                    "value": value,
                }
            )

    source = alt.InlineData(values=source)

    nearest = alt.selection(
        type="single", nearest=True, on="mouseover", fields=["epoch"], empty="none"
    )

    line = (
        alt.Chart(source)
        .mark_line()
        .encode(
            alt.X("epoch:O"),
            alt.Y("value:Q", axis=alt.Axis(format="%")),
            color="serie:N",
            tooltip=["epoch:O", "value:Q", "serie:N"],
        )
    )

    selectors = (
        alt.Chart(source)
        .mark_point()
        .encode(
            x="epoch:O",
            opacity=alt.value(0),
        )
        .add_selection(nearest)
    )

    # Draw points on the line, and highlight based on selection
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    # Draw text labels near the points, and highlight based on selection
    text = line.mark_text(align="left", dx=10, dy=-10).encode(
        text=alt.condition(nearest, "value:Q", alt.value(" ")),
    )

    # Draw a rule at the location of the selection
    rules = (
        alt.Chart(source)
        .mark_rule(color="gray")
        .encode(
            x="epoch:O",
        )
        .transform_filter(nearest)
    )

    # Put the five layers into a chart and bind the data
    chart = alt.layer(line, selectors, points, rules, text).properties(
        width="container", height=300
    )

    ctrl.chart_update(chart)
    state.flush("acc_loss")
