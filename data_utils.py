import torch


def dcu_numpy(x):
    return x.detach().cpu().numpy()