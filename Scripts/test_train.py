import numpy as np
import os
import sys
import torch
import json
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as tt

sys.path.insert(0, "C:/Users/Admin/Desktop/SHRG-Net")

from Logs.logger import logger


class HandDataset(Dataset):
    def __init__(self, data, img_size=256, heatmap_size=64):
        super().__init__()
        self.data = data
        self.data_keys = list(data.keys())
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        
        self.transform = tt.Compose([
            tt.Resize((img_size, img_size)),
            tt.ToTensor()
        ])

    def __len__(self):
        return len(self.data_keys)

    def __getitem__(self, idx):
        key = self.data_keys[idx]
        sample = self.data[key]

        imge = Image.open(sample["image"]).convert("RGB")
        X = self.transform(imge)

        ann = sample["annotation"]

        joints_raw = ann.get("joint_img", [])
        valid_raw = ann.get("joint_valid", [])

        joints = torch.tensor(joints_raw, dtype=torch.float32)
        valid_mask = torch.tensor(valid_raw, dtype=torch.float32)

        if joints.numel() > 0:
            scale_factor = self.heatmap_size / self.img_size
            joint_scaled = joints[:, :2] * scale_factor
        else:
            joint_scaled = torch.zeros((42, 2), dtype=torch.float32)
#            valid_mask = torch.zeros((21,), dtype=torch.float32)

        target_heatmaps = generate_heatmap(joint_scaled, valid_mask, output_size=(self.heatmap_size, self.heatmap_size))
        logger.info(valid_mask)
        valid_mask_expanded = valid_mask.view(-1, 1, 1)

        logger.info(valid_mask_expanded)

        return X, target_heatmaps, valid_mask_expanded




def generate_heatmap(joints, valid_mask, output_size=(64, 64), sigma=2.0):
    num_joints = joints.size(0)
    h, w = output_size
    heatmap = torch.zeros((num_joints, h, w), dtype=torch.float32)

    for i in range(num_joints):
        if valid_mask[i] <= 0:
            continue
        
        mu_x, mu_y = joints[i][0].item(), joints[i][1].item()

        if mu_x < 0 or mu_y < 0 or mu_x >= w or mu_y >= h:
            continue

        x = torch.arange(0, w, dtype=torch.float32)
        y = torch.arange(0, h, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        # Fixed: xx instead of x
        g = torch.exp(-((xx - mu_x) ** 2 + (yy - mu_y) ** 2) / (2 * sigma ** 2))
        heatmap[i] = g

    return heatmap


'''
train_data = HandDataset(data=dataset_dict)

train_loader = DataLoader(
    train_data,
    batch_size=10,
    shuffle=True
)

data_iter = iter(train_loader)
print(next(data_iter))'''