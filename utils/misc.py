import os
import errno
import random
import torch
import numpy as np
import subprocess
from .comm import is_main_process


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def set_seed(seed):
    print("set seed ",seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_config(cfg, path):
    if is_main_process():
        with open(path, 'w') as f:
            f.write(cfg.dump())


def to_device(targets, device):
    transfer_keys = set(['actioness', 'start_heatmap', 'end_heatmap', 'boxs', 'iou_map', 'candidates'])
    for idx in range(len(targets)):
        for key in targets[idx].keys():
            if key in transfer_keys:
                targets[idx][key] = targets[idx][key].to(device)
    return targets

def generate_gauss_weight(props_len, center, width):
        # pdb.set_trace()
    sigma = 9
    weight = torch.linspace(0, 1, props_len)
    weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
    center = center.unsqueeze(-1)
    width = width.unsqueeze(-1).clamp(1e-2) / sigma

    w = 0.3989422804014327
    weight = w/width*torch.exp(-(weight-center)**2/(2*width**2))

    return weight/weight.max(dim=-1, keepdim=True)[0]

def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss

def compute_project_term(mask_scores, gt_bitmasks, weight=None):
    mask_losses_y = dice_coefficient(
        mask_scores.max(dim=2, keepdim=True)[0],
        gt_bitmasks.max(dim=2, keepdim=True)[0]
    )
    mask_losses_x = dice_coefficient(
        mask_scores.max(dim=3, keepdim=True)[0],
        gt_bitmasks.max(dim=3, keepdim=True)[0]
    )
    return (mask_losses_x + mask_losses_y)

class NestedTensor(object):
    def __init__(self, tensors, mask, durations):
        self.tensors = tensors
        self.mask = mask
        self.durations = durations

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        cast_mask = self.mask.to(*args, **kwargs) if self.mask is not None else None
        return type(self)(cast_tensor, cast_mask, self.durations)

    def decompose(self):
        return self.tensors, self.mask, self.durations

    def subsample(self, stride, start_idx=0):
        # Subsample the video for multi-modal Interaction
        sampled_tensors = [video[start_idx::stride] for video in \
                            torch.split(self.tensors, self.durations, dim=0)]
        sampled_mask = [mask[start_idx::stride] for mask in \
                            torch.split(self.mask, self.durations, dim=0)]

        sampled_durations = [tensor.shape[0] for tensor in sampled_tensors]
        
        return NestedTensor(torch.cat(sampled_tensors,dim=0),
                        torch.cat(sampled_mask,dim=0), sampled_durations)

    @classmethod
    def from_tensor_list(cls, tensor_list):
        assert tensor_list[0].ndim == 4  # videos
        max_size = tuple(max(s) for s in zip(*[clip.shape for clip in tensor_list]))
        _, c, h, w = max_size
       
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
       
        # total number of frames in the batch
        durations = [clip.shape[0] for clip in tensor_list]
        nb_images = sum(clip.shape[0] for clip in tensor_list)  
        tensor = torch.zeros((nb_images, c, h, w), dtype=dtype, device=device)
        mask = torch.ones((nb_images, h, w), dtype=torch.bool, device=device)
        cur_dur = 0
        for i_clip, clip in enumerate(tensor_list):
            tensor[
                cur_dur : cur_dur + clip.shape[0],
                : clip.shape[1],
                : clip.shape[2],
                : clip.shape[3],
            ].copy_(clip)
            mask[
                cur_dur : cur_dur + clip.shape[0], : clip.shape[2], : clip.shape[3]
            ] = False
            cur_dur += clip.shape[0]

        return cls(tensor, mask, durations)

    def __repr__(self):
        return repr(self.tensors)