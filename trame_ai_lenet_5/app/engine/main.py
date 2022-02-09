import os
from pathlib import Path
import asyncio
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from trame import state

from . import ml, utils

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
