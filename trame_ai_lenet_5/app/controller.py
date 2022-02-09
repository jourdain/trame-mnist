from trame import controller as ctrl
from . import engine


def on_start():
    ctrl.on_ready = engine.initialize
    ctrl.start_training = engine.run_training
    ctrl.reset_training = engine.reset_training
