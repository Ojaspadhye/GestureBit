import os
import json

IMAGE_DIR = "C:/Users/Admin/Desktop/SHRG-Net/Data/images/InterHand2.6M_5fps_batch1/images/test/Capture1/ROM05_RT_Wrist_ROM/cam400283"
JSON_DATA = "C:/Users/Admin/Desktop/SHRG-Net/Data/InterHand2.6M_train_data.json"
JSON_DATA1 = "C:/Users/Admin/Desktop/SHRG-Net/Data/InterHand2.6M_train_joint_3d.json"

## BBo => bounding bo oordinate scaling and croping the full resolution image
## Joint Valid => Suggests if the joints can or annot be used
## hand_type => left for left hand ||  right for right hand || interacting for both hands
## hand_type_valid => If or no can we use the data

'''
=====================================
<class 'dict'>
dict_keys(['images', 'annotations'])
24004
{'id': 24004, 'image_id': 24004, 'bbox': [68.95188903808594, 123.58688354492188, 193.60272216796875, 194.743896484375], 'joint_valid': [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]], 'hand_type': 'interacting', 'hand_type_valid': 1}
<class 'int'>
24004
=====================================
'''


images = os.listdir(IMAGE_DIR)

with open(JSON_DATA, "r") as f:
    data = json.load(f)

image_id = []
dict = {}
for i, image in enumerate(images):
    ids = image.replace("image", "").replace(".jpg", "")
    image_id.append(ids)

    for k, j in enumerate(data["annotations"]):
        if j["id"] == int(image_id[ids]):
            print(i)
            print(j)
            print(type(j["id"]))



print("=====================================")

print(type(data))
print(data.keys())




print(image_id[0])

print("=====================================")


#print(dict(list(data.items()))[0])
#print(image_id)
#print(data["100"])

