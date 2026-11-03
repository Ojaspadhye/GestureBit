## imports
import torch
from tqdm import tqdm
from torch.optim import SGD, lr_scheduler
import sys
import time
import os, json
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "C:/Users/Admin/Desktop/SHRG-Net")

from Logs.logger import logger
from src.model.StackedHourGlass import StackedHourGlassBlock
from src.model.lossfunction import MSELossFuntion
from Scripts.test_train import HandDataset, generate_heatmap

## DEvice check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

start_time = time.perf_counter()

print("==========================")
print(f"Devie used: {device}")
print(f"Start Time: {time.ctime(time.time())}")
logger.info(f"Eperiment started {time.ctime(time.time())}")
print("==========================")
print("")

IMAGE_DIR = "C:/Users/Admin/Desktop/SHRG-Net/Data/test1_image/train"
JSON_DATA = "C:/Users/Admin/Desktop/SHRG-Net/Data/InterHand2.6M_train_data.json"


print("==========================")
print("Colletion and mathing of images to annotations")
##
images = os.listdir(IMAGE_DIR)
with open(JSON_DATA, "r") as f:
    json_data = json.load(f)

dataset_dict = {}
idx_counter = 0

for i, image in tqdm(enumerate(images)):
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



print("==========================")
print("")
print("==========================")
print("Training Initiated")
print("==========================")

train_data = HandDataset(data=dataset_dict)

train_loader = DataLoader(
    train_data,
    batch_size=10,
    shuffle=True
)

## Model Bit

model = StackedHourGlassBlock().to(device)
optimiser = SGD(params=model.parameters(), lr=0.001, momentum=0.9)
shedular = lr_scheduler.ExponentialLR(optimizer=optimiser, gamma=0.9)
loss_func = MSELossFuntion()
#train_data = DataLoader(train_data)

epoh_loss_list = []


epohes = 80
for epoh in tqdm(range(epohes)):
    model.train()
    running_loss = 0.0
    total_sample = 0

    for batch_idx, (inputs, targets, valid_mask) in tqdm(enumerate(train_loader)):

        inputs = inputs.to(device)
#        logger.info(type(inputs))
#        logger.info(f"inputs: {inputs.shape}")
        targets = targets.to(device)
        valid_mask = valid_mask.to(device)

        logger.info(valid_mask)

        optimiser.zero_grad()

        outputs = model(inputs)
        logger.info(type(outputs))
        logger.info(f"outputs: {outputs.shape}")

        logger.info(type(targets))
        logger.info(f"targets: {targets.shape}")

#        logger.info("")
#        logger.info(targets)
#        logger.info("===========================================")
#        logger.info(outputs)

        loss = 0.0
        for out_stack in outputs:
#            logger.info(out_stack, targets, valid_mask)
            loss += loss_func(out_stack, targets, valid_mask)


        loss.backward()
        optimiser.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        total_sample += batch_size  # Fixed typo: sie -> size

    epoh_loss = running_loss / total_sample
    epoh_loss_list.append(epoh_loss)



print("")
print("==========================")
print(f"Model Name: {model.__class__.__name__}")
print(f"Number of Epohes: {epohes}")
print(f"Optimiser Used: {optimiser.__class__.__name__}")
print(f"Loss Funtion: {loss_func.__class__.__name__}")
print("Firt 50 epoh ll 0.001")
print("last 30 epoh ll 0.0001")
print("==========================")
print("")
print

print("==========================")
print(f"Sum Loss: {sum(epoh_loss_list)}")
print(f"Loss: {epoh_loss_list}")
print("==========================")

torch.save(model.state_dict(), "model_weights.pth")
torch.save(model, "omplete_model.pth")