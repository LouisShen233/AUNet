import os
import torch
from .aunet import AUNet


def build_update_module(cfg):
    model = AUNet(cfg)

    weights = torch.load(cfg["UPDATE"]["CHECKPOINT_PATH"])['state_dict']
    model.load_state_dict(weights)
    model.eval().cuda()
    return model

