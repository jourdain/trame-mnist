import os
import base64
import random
import asyncio
from pathlib import Path

import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor

from . import ml, utils, charts
from trame import state, controller as ctrl

MULTI_PROCESS_MANAGER = multiprocessing.Manager()
PROCESS_EXECUTOR = ProcessPoolExecutor(1)
PENDING_TASKS = []

# -----------------------------------------------------------------------------
# Initial state
# -----------------------------------------------------------------------------

TRAINING_INITIAL_STATE = {
    "epoch_end": 0,
    "model_state": {
        "epoch": 0,
        "training_accuracy": [],
        "training_loss": [],
        "validation_accuracy": [],
        "validation_loss": [],
    },
}

state.update(
    {
        **TRAINING_INITIAL_STATE,
        "training_running": False,
        "prediction_success": False,
        "prediction_available": False,
    }
)


def initialize(**kwargs):
    if ml.has_trained_model() and state.epoch_end == 0:
        # Just load existing state
        asyncio.create_task(training_add())

    state.prediction_available = ml.prediction_reload()
    prediction_update()


# -----------------------------------------------------------------------------
# Methods to bound to UI
# -----------------------------------------------------------------------------


async def training_add():
    """Add 10 epoch to current training"""
    await asyncio.gather(*PENDING_TASKS)
    PENDING_TASKS.clear()

    if state.model_state.get("epoch") >= state.epoch_end:
        state.epoch_end += 10

    loop = asyncio.get_event_loop()
    queue = MULTI_PROCESS_MANAGER.Queue()

    task_training = loop.run_in_executor(
        PROCESS_EXECUTOR,
        partial(ml.training_add, queue, state.epoch_end),
    )
    task_monitor = loop.create_task(utils.queue_to_state(queue, task_training))

    # Only join on monitor task
    PENDING_TASKS.append(task_monitor)

    state.prediction_available = ml.prediction_reload()


# -----------------------------------------------------------------------------


def training_reset():
    """Remove saved model and reset local state"""
    ml.delete_model()
    state.update(TRAINING_INITIAL_STATE)
    state.prediction_available = ml.prediction_reload()


# -----------------------------------------------------------------------------


def prediction_update():
    image, label, prediction = ml.prediction_update()

    image_path = Path(f"{ml.DATA_DIR}/{label}.jpg")
    image.save(image_path)
    with open(image_path, "rb") as file:
        data = base64.encodebytes(file.read()).decode("utf-8")
        state.prediction_input_url = f"data:image/jpeg;base64,{data}"

    state.prediction_label = label
    state.prediction_success = max(prediction) == prediction[label]
    ctrl.chart_pred_update(charts.prediction_chart(prediction))


# -----------------------------------------------------------------------------


async def prediction_next_failure():
    prediction_update()
    state.flush("prediction_success")  # Force it to be green

    while state.prediction_success:
        with state.monitor():
            prediction_update()
        await asyncio.sleep(0.05)


# -----------------------------------------------------------------------------


def xai_run():
    model, image = ml.prediction_xai_params()
    result = ml.xai_update(model, image)
    print(result)
    print("=" * 60)
    print(dir(result))


# -----------------------------------------------------------------------------
# State listeners
# -----------------------------------------------------------------------------


@state.change("model_state")
def update_charts(model_state, **kwargs):
    acc, loss = charts.acc_loss_charts(model_state)
    ctrl.chart_acc_update(acc)
    ctrl.chart_loss_update(loss)
