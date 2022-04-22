import os
import cv2
import random
from keras import layers
import keras.models
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
# height, width = loaded_img.shape[:2]
print(loaded_img.shape)
# print(width)
cv2.imshow("test", loaded_img)
cv2.waitKey(5000)

    # probability = model.predict(cv2) # is 4624,3468     needed 480, 640