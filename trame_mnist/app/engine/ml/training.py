import torch
import torchvision

from .common import TRANSFORM, DATA_DIR, get_model

DATASET_TRAINING = torchvision.datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=TRANSFORM,
)


def create_training_loaders(batch_size):
    training_set = DATASET_TRAINING

    training_set, validation_set = torch.utils.data.random_split(
        training_set, [55000, 5000]
    )

    train_loader = torch.utils.data.DataLoader(
        training_set, batch_size=batch_size, shuffle=True
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=True,
    )

    return train_loader, validation_loader


# -----------------------------------------------------------------------------


def training_add(queue, end_epoch, learning_rate=1e-5, batch=32):
    queue.put_nowait(dict(training_running=True, epoch_end=end_epoch))
    model = get_model(learning_rate)
    training_loader, validation_loader = create_training_loaders(batch)

    while model.epoch < end_epoch:
        model.train_step(training_loader)
        model.validation_step(validation_loader)
        model.epoch += 1
        queue.put_nowait({"model_state": model.metadata})

    queue.put_nowait(
        {
            "epoch_end": max(end_epoch, model.epoch),
            "model_state": model.metadata,
            "training_running": False,
        }
    )

    model.save()
    queue.put_nowait("stop")
