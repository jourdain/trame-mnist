import os
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------

DATA_DIR = str(
    Path(Path(__file__).parent.parent.parent.parent, "data").resolve().absolute()
)

MODEL_PATH = Path(DATA_DIR, "model_lenet-5.trained").resolve().absolute()

TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0,), (1,))]
)

# -----------------------------------------------------------------------------


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.c1 = nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2
        )
        self.c2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0
        )
        self.c3 = nn.Conv2d(
            in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=10)

    def forward(self, img):
        x = self.c1(img)
        x = self.relu(self.max_pool(x))
        x = self.c2(x)
        x = self.relu(self.max_pool(x))
        x = self.relu(self.c3(x))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -----------------------------------------------------------------------------


class Model:
    def __init__(self, model, learning_rate=1e-5):
        self.model = model
        self.lr = learning_rate
        self.loss = nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.epoch = 0

    def eval(self, *args, **kwds):
        return self.model.eval(*args, **kwds)

    def __call__(self, *args, **kwds):
        return self.model(*args, **kwds)

    def batch_accuracy(self, output, target):
        # output shape: [batch, 10]
        output = nn.functional.softmax(output, dim=1)
        output = output.argmax(1)
        acc = torch.sum(output == target) / output.shape[0]
        return float(acc)

    def train_step(self, dataset):
        self.model.train()
        batch_loss = []
        batch_acc = []
        for inputs, targets in dataset:
            outputs = self.model(inputs)

            loss = self.loss(outputs, targets)
            loss.backward()
            self.opt.step()
            batch_loss.append(loss.item())
            batch_acc.append(self.batch_accuracy(outputs, targets))

        self.train_loss.append(float(np.mean(batch_loss)))
        self.train_acc.append(float(np.mean(batch_acc)))

    def validation_step(self, dataset):
        self.model.eval()
        batch_loss = []
        batch_acc = []
        for inputs, targets in dataset:
            outputs = self.model(inputs)

            loss = self.loss(outputs, targets)
            batch_loss.append(loss.item())
            batch_acc.append(self.batch_accuracy(outputs, targets))

        self.val_loss.append(float(np.mean(batch_loss)))
        self.val_acc.append(float(np.mean(batch_acc)))

    def load(self, model_path=MODEL_PATH):
        if Path(model_path).exists():
            data = torch.load(model_path)
            self.metadata = data.get("metadata")
            self.model.load_state_dict(data.get("state_dict"))
            self.model.eval()

    def save(self, output_path=MODEL_PATH):
        data = {
            "state_dict": self.model.state_dict(),
            "metadata": self.metadata,
        }
        torch.save(data, output_path)

    def predict(self, image):
        self.model.eval()
        tensor = TRANSFORM(image)
        tensor = torch.reshape(tensor, (1, *tensor.shape))
        return self.model(tensor)

    @property
    def metadata(self):
        return {
            "training_accuracy": self.train_acc,
            "training_loss": self.train_loss,
            "validation_accuracy": self.val_acc,
            "validation_loss": self.val_loss,
            "epoch": self.epoch,
        }

    @metadata.setter
    def metadata(self, value):
        self.train_acc = value.get("training_accuracy")
        self.train_loss = value.get("training_loss")
        self.val_acc = value.get("validation_accuracy")
        self.val_loss = value.get("validation_loss")
        self.epoch = value.get("epoch")


# -----------------------------------------------------------------------------
# API to be used outside
# -----------------------------------------------------------------------------


def get_model(learning_rate=1e-5):
    lenet5 = LeNet5()
    model = Model(lenet5, learning_rate)
    model.load()
    return model


def delete_model():
    if MODEL_PATH.exists():
        os.remove(MODEL_PATH)


def has_trained_model():
    return MODEL_PATH.exists()
