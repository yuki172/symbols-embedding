import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from lightly.data import LightlyDataset
from lightly.transforms import SimCLRTransform  # type: ignore
import argparse

from model import SimCLRModel
from pytorch_lightning.callbacks import Callback
import json


class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train_loss_ssl")
        if loss is not None:
            self.epoch_losses.append(loss.item())


def train(max_epochs, batch_size, input_size, learning_rate):
    transform = SimCLRTransform(input_size=input_size, vf_prob=0.5, rr_prob=0.5)

    path_to_data = "data/train"
    dataset_train_simclr = LightlyDataset(input_dir=path_to_data, transform=transform)  # type: ignore

    dataloader_train_simclr = torch.utils.data.DataLoader(
        dataset_train_simclr,  # type: ignore
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )

    model = SimCLRModel(max_epochs=max_epochs)

    loss_history = LossHistory()
    trainer = pl.Trainer(
        max_epochs=max_epochs, devices=1, accelerator="auto", callbacks=[loss_history]
    )
    trainer.fit(model, dataloader_train_simclr)

    name = f"model_{batch_size}_{input_size}_{learning_rate}_{max_epochs}"

    trainer.save_checkpoint(f"saved/{name}.ckpt")

    trainer.logged_metrics["train_loss_ssl"]

    os.makedirs("metrics", exist_ok=True)

    with open(f"metrics/{name}.json", "w") as file:
        json.dump({"losses": loss_history.epoch_losses}, file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # training
    parser.add_argument(
        "--max_epochs",
        type=int,
        dest="max_epochs",
        default=1,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        dest="batch_size",
        default=16,
        help="batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        dest="learning_rate",
        default=6e-2,
        help="learning rate",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        dest="input_size",
        default=128,
        help="input shape",
    )

    args = parser.parse_args()

    train(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        input_size=args.input_size,
        learning_rate=args.learning_rate,
    )
