import json
import cv2
import os

with open("/home/ranais/Ranai/Research/object-detection-ex-template/Training/annotation-20220413T205639Z-001/annotation/final_anns.json", "r") as annotn_file:
    annotn = json.load(annotn_file)

root_dir = "/home/ranais/Ranai/Research/object-detection-ex-template/Training/frames-20220412T154427Z-001/frames"
imgs = os.listdir(root_dir)
training_image_set = []
training_annotn_set = []
for img in imgs:
    training_image_set.append(cv2.imread(root_dir+"/"+img))
    found_duck = False
    for dict_item in annotn[img]:
        if dict_item["cat_id"] == 2:
            training_annotn_set.append(1)
            found_duck = True
            break
    if not found_duck:
        training_annotn_set.append(0)

# for i in range(9):
#     cv2.imshow(f"{imgs[i]} + {training_annotn_set[i]}", training_image_set[i])
#     cv2.waitKey(1000)
