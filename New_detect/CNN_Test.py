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
# print(img_list)
model = keras.models.load_model('/home/ranais/Downloads/CNN_Detect_model.h5')

# for i in img_list:
pic_loc = pic_dir + "/" + random.choice(img_list)

loaded_img = cv2.imread(pic_loc)
print(loaded_img.shape)
loaded_array = np.array(loaded_img)
print(loaded_array.shape)

opencv = cv2.resize(loaded_img, (480, 640))
print(opencv.shape)
opencv_array = np.array(opencv)
print(opencv_array.shape)

keras = keras.preprocessing.image.img_to_array(loaded_img.resize(480, 640)) # This line is not reszing the image into 4D space.
print(opencv.shape)
keras = keras[:, :, :, 1]

print(model.predict(opencv_array))

# # cv2.imshow("test", loaded_img)  # TODO Why is imshow resizing it. Resize it to 480 by 640 for the CNN. Try with PIL.



    # probability = model.predict(cv2) # is 4624,3468     needed 480, 640