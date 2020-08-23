# encoding: utf-8


import torch


def train_collate_fn(batch):
    imgs, pids, _, _, mask_target, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    mask_target = torch.tensor(mask_target, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, mask_target
    
def clustering_collate_fn(batch):
    imgs, pids, _, _, _, mask_target_path = zip(*batch)
    return torch.stack(imgs, dim=0), mask_target_path, pids


def val_collate_fn(batch):
    imgs, pids, camids, _, mask_target, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids