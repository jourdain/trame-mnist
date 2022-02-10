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


MODEL_PATH = Path(ml.DATA_DIR, "model_lenet-5.trained").resolve().absolute()
MULTI_PROCESS_MANAGER = multiprocessing.Manager()
PENDING_TASKS = []
CURRENT_INPUT = None
CURRENT_LABEL = 0
ML_MODEL = None

# -----------------------------------------------------------------------------
# Initial state
# -----------------------------------------------------------------------------

state.update(
    {
        "epoch_end": 0,
        "training_running": False,
        "prediction_success": False,
        "model_state": {
            "epoch": 0,
            "training_accuracy": [],
            "training_loss": [],
            "validation_accuracy": [],
            "validation_loss": [],
        },
        "prediction_results": [],
    }
)


def initialize(**kwargs):
    if MODEL_PATH.exists() and state.epoch_end == 0:
        asyncio.create_task(run_training())
    update_prediction_input()


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

    global ML_MODEL
    ML_MODEL = None


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

    global ML_MODEL
    ML_MODEL = None


# -----------------------------------------------------------------------------


def update_prediction_input():
    global CURRENT_INPUT, CURRENT_LABEL
    ds = ml.TEST_DATASET
    size = len(ds)
    image, label_nb = ds[random.randint(0, size - 1)]
    image_path = Path(f"{ml.DATA_DIR}/{label_nb}.jpg")
    image.save(image_path)
    with open(image_path, "rb") as file:
        data = base64.encodebytes(file.read()).decode("utf-8")
        state.prediction_input_url = f"data:image/jpeg;base64,{data}"

    CURRENT_INPUT = image
    CURRENT_LABEL = label_nb
    state.prediction_label = label_nb
    run_prediction()


# -----------------------------------------------------------------------------


def get_prediction_model():
    global ML_MODEL
    if ML_MODEL is not None:
        return ML_MODEL

    ML_MODEL = ml.get_trained_model(MODEL_PATH)
    return ML_MODEL


# -----------------------------------------------------------------------------


def run_prediction():
    model = get_prediction_model()

    if model:
        result = model.predict(CURRENT_INPUT)
        result = result[0].tolist()

        ctrl.chart_pred_update(charts.prediction_chart(result))
        state.prediction_results = result
        state.prediction_success = max(result) == result[CURRENT_LABEL]


# -----------------------------------------------------------------------------


async def find_next_fail():
    update_prediction_input()

    while state.prediction_success:
        with state.monitor():
            update_prediction_input()
        await asyncio.sleep(0.05)


# -----------------------------------------------------------------------------
# State listeners
# -----------------------------------------------------------------------------


@state.change("model_state")
def update_charts(model_state, **kwargs):
    acc, loss = charts.acc_loss_charts(model_state)
    ctrl.chart_acc_update(acc)
    ctrl.chart_loss_update(loss)
