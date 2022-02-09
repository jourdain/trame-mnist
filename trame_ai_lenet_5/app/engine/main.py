import os
import asyncio
from pathlib import Path

import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor

from . import ml, utils, charts
from trame import state, controller as ctrl


MODEL_PATH = Path(ml.DATA_DIR, "model_lenet-5.trained").resolve().absolute()
MULTI_PROCESS_MANAGER = multiprocessing.Manager()
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


def initialize():
    if MODEL_PATH.exists() and state.epoch_end == 0:
        asyncio.create_task(run_training())


# -----------------------------------------------------------------------------
# Methods to bound to UI
# -----------------------------------------------------------------------------


async def run_training():
    """Add 10 epoch to current training"""
    await asyncio.gather(*PENDING_TASKS)
    PENDING_TASKS.clear()

    if state.model_state.get("epoch") >= state.epoch_end:
        state.epoch_end += 10

    loop = asyncio.get_event_loop()
    queue = MULTI_PROCESS_MANAGER.Queue()
    task_training = loop.run_in_executor(
        ProcessPoolExecutor(1),
        partial(ml.train_model, MODEL_PATH, queue, state.epoch_end),
    )
    task_monitor = loop.create_task(utils.queue_to_state(queue, task_training))

    # Only join on monitor task
    PENDING_TASKS.append(task_monitor)


# -----------------------------------------------------------------------------


def reset_training():
    """Remove saved model and reset local state"""
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


# -----------------------------------------------------------------------------
# State listeners
# -----------------------------------------------------------------------------


@state.change("model_state")
def update_charts(model_state, **kwargs):
    acc, loss = charts.acc_loss_charts(model_state)
    ctrl.chart_acc_update(acc)
    ctrl.chart_loss_update(loss)


# -----------------------------------------------------------------------------
# Debug to check that server is not busy...
# -----------------------------------------------------------------------------


@state.change("slider_value")
def monitor_slider(slider_value, **kwargs):
    print(slider_value)
