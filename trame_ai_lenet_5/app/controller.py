from trame import controller as ctrl
from . import engine


def on_start():
    ctrl.on_ready = engine.initialize

    ctrl.training_add = engine.run_training
    ctrl.training_reset = engine.reset_training

    ctrl.prediction_update_input = engine.update_prediction_input
    ctrl.prediction_run = engine.run_prediction
