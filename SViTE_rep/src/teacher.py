"""Load the DeiT authors' pretrained ImageNet-1k convolutional teacher."""

import timm
import torch
from torch import nn


TEACHER_MODEL = "regnety_160.deit_in1k"


def build_teacher(device: str | torch.device = "cpu") -> nn.Module:
    """Download/cache the authors' weights on first call, then freeze the model.

    The explicit .deit_in1k tag matters: another RegNetY checkpoint is not
    necessarily the teacher used in this paper.
    """
    teacher = timm.create_model(TEACHER_MODEL, pretrained=True, num_classes=1000)
    return teacher.to(device).requires_grad_(False).eval()
