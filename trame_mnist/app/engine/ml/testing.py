import torch
import torchvision
import torch.nn as nn
import numpy as np
import time

from .common import DATA_DIR, TRANSFORM
from .prediction import get_model

BATCH_SIZE = 32

DATASET_TEST = torchvision.datasets.MNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=TRANSFORM,
)

LOADER_TEST = torch.utils.data.DataLoader(
    DATASET_TEST,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


def winner_class(classes):
    v = np.amax(classes)
    i = np.where(classes == v)
    return int(i[0])


@torch.no_grad()
def testing_run(datasets=LOADER_TEST):
    model = get_model().model
    model.eval()
    confusion_matrix = np.zeros((10, 10), dtype=np.float64)
    for inputs, targets in datasets:
        outputs = model(inputs).numpy()
        for i in range(inputs.shape[0]):
            confusion_matrix[winner_class(outputs[i])][int(targets[i])] += 1

    # Normalize
    total = np.sum(confusion_matrix)
    for i in range(10):
        ratio = np.sum(confusion_matrix[i])
        confusion_matrix[i] *= 100 / ratio

    return np.around(confusion_matrix), int(total)
