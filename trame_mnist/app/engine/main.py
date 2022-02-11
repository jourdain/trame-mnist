import numpy as np
import base64
import asyncio
from pathlib import Path

import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor

from . import ml, utils, charts
from trame import state, controller as ctrl

MULTI_PROCESS_MANAGER = None
PROCESS_EXECUTOR = None
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
    "xai_results": [],
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
    global MULTI_PROCESS_MANAGER, PROCESS_EXECUTOR
    MULTI_PROCESS_MANAGER = multiprocessing.Manager()
    PROCESS_EXECUTOR = ProcessPoolExecutor(1)

    if ml.has_trained_model() and state.epoch_end == 0:
        # Just load existing state
        asyncio.create_task(training_add())

    reset_model()
    prediction_update()


# -----------------------------------------------------------------------------
# Methods to bound to UI
# -----------------------------------------------------------------------------

state.epoch_increase = 2  # we need a minimum of 2 points to plot progress


async def training_add():
    """Add 10 epoch to current training"""
    await asyncio.gather(*PENDING_TASKS)
    PENDING_TASKS.clear()

    if state.model_state.get("epoch") >= state.epoch_end:
        state.epoch_end += state.epoch_increase

    loop = asyncio.get_event_loop()
    queue = MULTI_PROCESS_MANAGER.Queue()

    task_training = loop.run_in_executor(
        PROCESS_EXECUTOR,
        partial(ml.training_add, queue, state.epoch_end),
    )
    task_monitor = loop.create_task(utils.queue_to_state(queue, task_training))

    # Only join on monitor task
    PENDING_TASKS.append(task_monitor)

    reset_model()


# -----------------------------------------------------------------------------


def training_reset():
    """Remove saved model and reset local state"""
    ml.delete_model()
    state.update(TRAINING_INITIAL_STATE)
    reset_model()


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

    if state.xai_viz:
        xai_run()


# -----------------------------------------------------------------------------


def _prediction_next_failure():
    with state.monitor():
        prediction_update()
        if not state.prediction_success:
            state.prediction_search_failure = False

    if state.prediction_success and state.prediction_search_failure:
        loop = asyncio.get_event_loop()
        loop.call_later(0.01, _prediction_next_failure)


# -----------------------------------------------------------------------------


def xai_run():
    try:
        results = {}
        model, image = ml.prediction_xai_params()
        for xai_method in ml.SALIENCY_TYPES:
            result = ml.xai_update(model, image, xai_method)
            heatmaps = {}
            data_range = [float(np.amin(result)), float(np.amax(result))]
            for i in range(10):
                heatmaps[f"{i}"] = result[i].ravel().tolist()
            results[xai_method] = {"heatmaps": heatmaps, "range": data_range}

        state.xai_results = results
    except Exception:
        pass  # Model is not available...


# -----------------------------------------------------------------------------


def _testing_running():
    with state.monitor():
        matrix, sample_size = ml.testing_run()
        ctrl.chart_confusion_matrix(charts.confusion_matrix_chart(matrix))
        ctrl.chart_class_accuracy(charts.class_accuracy(matrix))

        state.testing_count = sample_size
        state.testing_running = False


# -----------------------------------------------------------------------------


def testing_run():
    state.testing_running = True
    loop = asyncio.get_event_loop()
    loop.call_later(0.01, _testing_running)


# -----------------------------------------------------------------------------


def reset_model():
    state.prediction_available = ml.prediction_reload()
    state.testing_count = 0


# -----------------------------------------------------------------------------
# State listeners
# -----------------------------------------------------------------------------


@state.change("model_state")
def update_charts(model_state, **kwargs):
    acc, loss = charts.acc_loss_charts(model_state)
    ctrl.chart_acc_update(acc)
    ctrl.chart_loss_update(loss)


@state.change("xai_viz_color_min", "xai_viz_color_max")
def update_xai_color_range(xai_viz_color_min, xai_viz_color_max, **kwargs):
    state.xai_viz_color_range = [xai_viz_color_min, xai_viz_color_max]


@state.change("xai_viz")
def toggle_xai_viz(xai_viz, **kwargs):
    if xai_viz:
        xai_run()


@state.change("prediction_search_failure")
def toggle_search_failue(prediction_search_failure, **kwargs):
    if prediction_search_failure:
        _prediction_next_failure()
