from trame import controller as ctrl
from . import engine


def on_start():
    ctrl.start_training = engine.run_training
    ctrl.reset_training = engine.reset_training
