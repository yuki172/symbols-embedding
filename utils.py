import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from lightly.transforms import utils
from lightly.data import LightlyDataset
from sklearn.preprocessing import normalize
import json

from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl


def generate_embeddings(model, dataloader):
    """Generates representations for all images in the dataloader with
    the given model
    """
    model.eval()
    embeddings = []
    filenames = []
    with torch.no_grad():
        for img, _, fnames in dataloader:
            img = img.to(model.device)
            emb = model.backbone(img).flatten(start_dim=1)
            embeddings.append(emb)
            filenames.extend(fnames)

    embeddings = torch.cat(embeddings, 0)
    embeddings = normalize(embeddings)
    return embeddings, filenames
