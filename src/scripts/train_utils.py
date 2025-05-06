import math
import os
import shutil

import torch


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"] * 1e6


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm%ds" % (m, s)


class AverageMeter(object):
    """Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, name=None):
    os.makedirs("./checkpoints_path", exist_ok=True)
    if name is None:
        name = "detect_ai_model"

    filename = f"./checkpoints_path/{name}_last.pth.tar"
    torch.save(state, filename, _use_new_zipfile_serialization=False)

    if is_best:
        shutil.copyfile(filename, f"./checkpoints_path/{name}_best.pth.tar")
