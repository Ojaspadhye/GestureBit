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
from src.model.StackedHourGlass import StackedHourGlassBlock

IMAGE_DIR = "C:/Users/Admin/Desktop/SHRG-Net/Data/images/InterHand2.6M_5fps_batch1/images/test/Capture1/ROM05_RT_Wrist_ROM/cam400283"
JSON_DATA = "C:/Users/Admin/Desktop/SHRG-Net/Data/InterHand2.6M_train_data.json"

images = os.listdir(IMAGE_DIR)
with open(JSON_DATA, "r") as f:
    json_data = json.load(f)

# Build a properly structured dataset dictionary
dataset_dict = {}
idx_counter = 0

for image in images:
    # Adjust this parsing logic if your JSON IDs match differently
    ids = image.replace("image", "").replace(".jpg", "")
    if not ids.isdigit():
        continue
    img_id = int(ids)

    for j in json_data["annotations"]:
        if j.get("id") == img_id:
            dataset_dict[idx_counter] = {
                "image": os.path.join(IMAGE_DIR, image),
                "annotation": j
            }
            idx_counter += 1
            break


class HandDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.data_keys = list(data.keys())
        
        self.transform = tt.Compose([
            tt.Resize((256, 256)),
            tt.ToTensor()
        ])

    def __len__(self):
         return len(self.data_keys)

    def __getitem__(self, idx):
        key = self.data_keys[idx]
        sample = self.data[key]

        # 1. Open image using the file path string
        imge = Image.open(sample["image"]).convert("RGB")
        X = self.transform(imge)  # Fixed typo: self.transform instead of self.transforms

        # 2. Extract annotations safely
        ann = sample["annotation"]
        Y = {
            "bbox": torch.tensor(ann.get("box", [0, 0, 0, 0]), dtype=torch.float32), # adjust key name if needed
            "joint_valid": torch.tensor(ann.get("joint", []), dtype=torch.float32),
            "hand_type_valid": torch.tensor(ann.get("hand_type_valid", 0)),
        }

        return X, Y


class InterHandDataset(Dataset):
    def __init__(self, annotations, img_s = 256, heatmap_c=54, sigma=2):
        super().__init__()
        self.annotations = annotations
        self.img_s = img_s
        self.heatmap_c = heatmap_c
        self.joints = 42
        self.sigma = sigma

    def generate_heatmap(self, kp):
        heatmaps = np.zeros((self.jointes, self.heatmap_c, self.heatmap_c), dtype=np.float32)

        x = np.arange(0, self.heatmap_c, 1, np.float32)
        y = np.arange(0, self.heatmap_c, 1, np.float32)
        yy, xx = np.meshgrid(y, x, indexing='ij')

        for i in range(self.joints):
            pt = kp[i]
            if pt[0] < 0 or pt[1] < 0 or pt[0] >= self.heatmap_c:
                continue

            gaussian = np.exp(-(( - pt[0]) ** 2 + (yy - pt[1]) ** 2) / (2 * self.sigma ** 2))

        return torch.tensor(heatmaps, dtype=torch.float32)

'''
train_data = HandDataset(data=dataset_dict)

train_loader = DataLoader(
    train_data,
    batch_size=10,
    shuffle=True
)

data_iter = iter(train_loader)
print(next(data_iter))'''