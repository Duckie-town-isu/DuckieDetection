import os
import cv2
import random
from keras import layers
import keras.models
import numpy as np
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img

pic_dir = "/home/ranais/Ranai/Research/object-detection-ex-template/Training/Testing/Ducks-20220422T151442Z-001/Ducks"
img_list = os.listdir(pic_dir)
model = keras.models.load_model('/home/ranais/Downloads/CNN_Detect_model.h5')

for i in img_list:
    chosen_img = random.choice(img_list)
    pic_loc = pic_dir + "/" + chosen_img

    loaded_img = cv2.imread(pic_loc)

    resized_img = cv2.resize(loaded_img, (640, 480))

    img_array = np.array(resized_img)
    # img_array = img_array[:, :, 1]
    CNN_input = np.array([np.zeros((480, 640, 1))])
    CNN_input[0] = img_array[:, :, :1]

    # if model.predict(CNN_input)[0] > 0.4 and model.predict(CNN_input)[1]:
    #     cv2.imshow(f"Duck Found in img {chosen_img}", resized_img)
    # else:
    #     cv2.imshow(f"No Duck Found in img {chosen_img}", resized_img)