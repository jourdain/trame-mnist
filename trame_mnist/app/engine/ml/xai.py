from scipy.special import softmax

# pytorch
import torch
import torch.nn as nn

# xaitk-saliency
from smqtk_classifier import ClassifyImage
from xaitk_saliency.impls.gen_image_classifier_blackbox_sal import (
    rise,
    slidingwindow as sw,
)

# App specific
from .common import TRANSFORM


# SMQTK black-box classifier
class ClfModel(ClassifyImage):
    def __init__(self, model):
        self.model = model
        self._labels = list(range(10))

    def get_labels(self):
        return self._labels

    @torch.no_grad()
    def classify_images(self, image_iter):
        for img in image_iter:
            inp = TRANSFORM(img).unsqueeze(0)
            vec = self.model(inp).cpu().numpy().squeeze()
            out = softmax(vec)
            yield dict(zip(self.get_labels(), out))

    def get_config(self):
        # Required by a parent class.
        return {}


class ClassificationSaliency:
    def __init__(self, method):
        self._saliency = method
        self._model = None
        self._class_model = None

    def set_model(self, model):
        if self._model != model:
            self._model = model
            self._class_model = ClfModel(self._model)

    def run(self, input, *_):
        return self._saliency(input, self._class_model)


SALIENCY_TYPES = ["RISEStack", "SlidingWindowStack"]

METHOD_RISE = rise.RISEStack(n=200, s=8, p1=0.5, seed=1234, threads=4, debiased=True)
METHOD_SW = sw.SlidingWindowStack(window_size=[2, 2], stride=[1, 1], threads=4)

INSTANCES = {
    "RISEStack": ClassificationSaliency(METHOD_RISE),
    "SlidingWindowStack": ClassificationSaliency(METHOD_SW),
}


def xai_update(model, input, name="RISEStack"):
    xai_model = INSTANCES[name]
    xai_model.set_model(model)
    return xai_model.run(input)
