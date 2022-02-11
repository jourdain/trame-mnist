import numpy as np
from scipy.special import softmax

# pytorch
import torch
import torch.nn as nn

# xaitk-saliency
from xaitk_saliency.impls.gen_image_classifier_blackbox_sal.rise import RISEStack
from xaitk_saliency.impls.gen_image_classifier_blackbox_sal.slidingwindow import (
    SlidingWindowStack,
)

# xaitk-saliency
from smqtk_classifier import ClassifyImage

# App specific
from .common import TRANSFORM

SALIENCY_TYPES = {
    "RISEStack": {
        "_saliency": {
            "class": RISEStack,
        },
    },
    "SlidingWindowStack": {
        "_saliency": {
            "class": SlidingWindowStack,
        },
    },
}

SALIENCY_PARAMS = {
    "RISEStack": ["n", "s", "p1", "seed", "threads", "debiased"],
    "SlidingWindowStack": ["window_size", "stride", "threads"],
}

SALIENCY_PARAMS_DEFAULTS = {
    "RISEStack": {
        "n": 200,
        "s": 8,
        "p1": 0.5,
        "seed": 1234,
        "threads": 4,
        "debiased": True,
    },
    "SlidingWindowStack": {
        "window_size": [2, 2],
        "stride": [1, 1],
        "threads": 4,
    },
}


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


class Saliency:
    def __init__(self, model, name, params):
        self._model = model
        try:
            for key, value in SALIENCY_TYPES[name].items():
                constructor = value.get("class")
                param_keys = value.get("params", params.keys())
                setattr(self, key, constructor(**{k: params[k] for k in param_keys}))
        except:
            print(f"Could not find {name} in {list(SALIENCY_TYPES.keys())}")


class ClassificationSaliency(Saliency):
    def run(self, input, *_):
        return self._saliency(input, ClfModel(self._model))


def xai_update(model, input, name="RISEStack"):
    xai_model = ClassificationSaliency(
        model,
        name,
        SALIENCY_PARAMS_DEFAULTS[name],
    )
    return xai_model.run(input)
