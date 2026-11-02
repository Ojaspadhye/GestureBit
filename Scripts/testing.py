from PIL import Image
import numpy as np 
import sys
import torch
from pathlib import Path
from torchvision import transforms as tt

sys.path.insert(0, "C:/Users/Admin/Desktop/SHRG-Net")

from src.model.StackedHourGlass import StackedHourGlassBlock

device = torch.cuda.is_available()
print(device)

IMAGE_PATH = "C:/Users/Admin/Desktop/SHRG-Net/Data/images/InterHand2.6M_5fps_batch1/images/test/Capture1/ROM01_No_Interaction_2_Hand/cam400263/image25098.jpg"
model = StackedHourGlassBlock()


img = Image.open(IMAGE_PATH)

print("======= Image Disription =======")
print(f"Channels: ", img.getbands())
print(f"Image dimentions", img.getpixel((0, 0)))

tensor_funtion = tt.Compose(
    [
        tt.Resize((256, 256)),
        tt.ToTensor()
    ]
)

t_img = tensor_funtion(img).unsqueeze(0)
print(type(t_img))
print("======= Image tensor =======")
print(t_img.shape)

val = torch.stack(model(t_img), dim=0)
print(type(val[0]))
print("====== Output of model ========")
print(f"Image disription: {len(val)}")


#print(val)

out = val.detach().cpu()
out_2d = out.squeeze()

out_np = out_2d.numpy()

if out_np.max() != out_np.min():
    out_np = (out_np - out_np.min())

class_indices = (out_np * 255).astype(np.uint8)

img_up = Image.fromarray(class_indices, mode="L")

img.show()