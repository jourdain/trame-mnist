import random
import numpy as np
import torchvision

from .common import DATA_DIR, get_model, has_trained_model

MODEL = None

DATASET_TEST = torchvision.datasets.MNIST(
    root=DATA_DIR,
    train=False,
    download=True,
)

LAST_IMAGE = None


def prediction_reload():
    global MODEL
    MODEL = get_model() if has_trained_model() else None
    prediction_update()
    return has_trained_model()


def prediction_update():
    # Input
    size = len(DATASET_TEST)
    image, label = DATASET_TEST[random.randint(0, size - 1)]
    prediction = np.zeros(10).tolist()

    # Prediction
    if MODEL is not None:
        prediction = MODEL.predict(image)
        prediction = prediction[0].tolist()

    # keep track of last input
    global LAST_IMAGE
    LAST_IMAGE = image

    return image, label, prediction


def prediction_xai_params():
    return MODEL.model, np.asarray(LAST_IMAGE)
