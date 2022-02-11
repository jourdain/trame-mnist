from trame import controller as ctrl
from . import engine


def on_start():
    ctrl.on_ready = engine.initialize

    ctrl.training_add = engine.training_add
    ctrl.training_reset = engine.training_reset

    ctrl.prediction_update = engine.prediction_update
    ctrl.prediction_next_failure = engine.prediction_next_failure

    ctrl.xai_run = engine.xai_run
