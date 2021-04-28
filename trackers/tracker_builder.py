import os
import torch
from .basenet import SiamRPN, SiamRPNBIG, SiamRPNOTB, SiamRPNVOT

tracker_dir = 'trackers/pretrained/'

def build_tracker(cfg):
    model_name = cfg["MODEL"]["MODEL_NAME"]
    checkpoint_file = os.path.join(tracker_dir, '{}.model'.format(model_name))
    net = eval(model_name + '()')
    net.load_state_dict(torch.load(checkpoint_file))
    net.eval().cuda()
    return net