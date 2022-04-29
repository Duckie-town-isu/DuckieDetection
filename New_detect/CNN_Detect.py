import json
import os

import cv2
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential

# with open(
#         "Training/annotation-20220413T205639Z-001/annotation/final_anns.json",
#         "r") as annotn_file:
#     annotn = json.load(annotn_file)

# root_dir = "Training/frames-20220415T203426Z-001"

with open(
        "../Training/annotation-20220413T205639Z-001/annotation/final_anns.json",
        "r") as annotn_file:
    annotn = json.load(annotn_file)

root_dir = "../Training/frames-20220412T154427Z-001/frames"
imgs = os.listdir(root_dir)
training_image_set = []
training_annotn_set = []

for img in imgs:
    training_image_set.append(cv2.imread(root_dir + "/" + img))
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

training_image_array = np.array(training_image_set)
print(training_image_array.shape)
training_annotn_array = np.array(training_annotn_set)
training_image_array = training_image_array[:, :, :, 1]
# training_image_array.reshape((training_image_array.shape[0], 480, 640, 1))
print(training_image_array.shape)
cv2.imshow("img", training_image_array[179])
cv2.waitKey(1000)


def CNN(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(480, 640, 1)))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(64, (1, 1), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(Conv2D(128, (1, 1), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(Conv2D(256, (1, 1), activation='relu'))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(Conv2D(256, (1, 1), activation='relu'))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(1024, (3, 3), activation='relu'))
    model.add(Conv2D(512, (1, 1), activation='relu'))
    model.add(Conv2D(1024, (3, 3), activation='relu'))
    model.add(Conv2D(512, (1, 1), activation='relu'))
    model.add(Conv2D(1024, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1)

    model.save_weights('CNN_Detect.h5')
    model.save('CNN_Detect.h5')

CNN(training_image_array, training_annotn_array, None, None)


