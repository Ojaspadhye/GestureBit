## imports
import torch
from tqdm import tqdm
from torch.optim import SGD
import sys
import time
import os, json
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "C:/Users/Admin/Desktop/SHRG-Net")

from Logs.logger import logger
from src.model.StackedHourGlass import StackedHourGlassBlock
from src.model.lossfunction import MSELossFuntion
from Scripts.test_train import HandDataset

## DEvice check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

start_time = time.perf_counter()

print("==========================")
print(f"Devie used: {device}")
print(f"Start Time: {time.ctime(time.time())}")
logger.info(f"Eperiment started {time.ctime(time.time())}")
print("==========================")


IMAGE_DIR = "C:/Users/Admin/Desktop/SHRG-Net/Data/images/InterHand2.6M_5fps_batch1/images/test/Capture1/ROM05_RT_Wrist_ROM/cam400283"
JSON_DATA = "C:/Users/Admin/Desktop/SHRG-Net/Data/InterHand2.6M_train_data.json"

##
images = os.listdir(IMAGE_DIR)
with open(JSON_DATA, "r") as f:
    json_data = json.load(f)

dataset_dict = {}
idx_counter = 0

for image in images:
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

train_data = HandDataset(data=dataset_dict)

train_loader = DataLoader(
    train_data,
    batch_size=10,
    shuffle=True
)

##

model = StackedHourGlassBlock()
optimiser = SGD(params=model.parameters())
loss_func = MSELossFuntion()

epoh_loss_list = []


epohes = 1
for epoh in tqdm(range(epohes)):
    model.train()

    running_loss = 0.0
    total_sample = 0

    for bath, (input, targets) in enumerate(train_loader):
        optimiser.zero_grad()

        outputs = model(input)
        loss = loss_func(outputs, targets)

        loss.backward()

        optimiser.step()

        running_loss += loss.item() * input.size(0)
        total_sample += input.sie(0)

        if outputs.ndim > 1 and outputs.size(1) > 1:
            preds = torch.argmax(outputs, dim=1)

    epoh_loss = running_loss / total_sample
    epoh_loss_list.append(epoh_loss)



print("")
print("==========================")
print(f"Model Name: {model.__class__.__name__}")
print(f"Number of Epohes: {epohes}")
print(f"Optimiser Used: {optimiser.__class__.__name__}")
print(f"Loss Funtion: {loss_func.__class__.__name__}")
print("==========================")
print("")

