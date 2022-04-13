import cv2
import keras.models
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random


import matplotlib.pyplot as plt
# %matplotlib inline

from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img

import os
import random
import gc

nrows= 480  #150
ncolumns = 640 #150
channels = 3


def read_and_process_image(list_of_images):
    X = []
    y = []

    for image in list_of_images:
        try:
            X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows, ncolumns), interpolation=cv2.INTER_CUBIC))

            if 'duck' in image[16:] or 'two' in image[16:]:
                y.append(1)
            elif 'road' in image[16:]:
                y.append(0)
        except Exception as e:
            print(str(e))

    return X, y

columns = 5
model = keras.models.load_model('model_keras.h5')

pic_dir = "/home/ranais/Ranai/Research/object-detection-ex-template/Ducks"
img_list = os.listdir(pic_dir)
print(img_list)

img = random.choice(img_list)
pic_loc = pic_dir + "/" + img

pic_dir = mpimg.imread(pic_loc)
# plt.imshow(pic_dir)
# plt.show()

X_test, y_test = read_and_process_image(pic_dir[:5])
x = np.array(X_test)
test_datagen = ImageDataGenerator(rescale=1./255)

i = 0
text_labels = []

image = keras.preprocessing.image.load_img(pic_loc)
print(type(image))
input_arr = keras.preprocessing.image.img_to_array(image.resize((150, 150)))
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = model.predict(input_arr)
if predictions > 0.5:
    text_labels.append('duck found!')
else:
    text_labels.append('no duck ')

# plt.subplot(int(5/columns+1), columns, i+1)
plt.title('' + text_labels[i])
imgplot = plt.imshow(image)

# plt.figure(figsize=(30,20))
# for batch in test_datagen.flow(x, batch_size=1):
#     pred = model.predict(batch)
#     if pred>0.5:
#         text_labels.append('duck found!')
#     else:
#         text_labels.append('no duck ')
#     plt.subplot(int(5/columns+1),columns, i+1)
#     plt.title(''+text_labels[i])
#     imgplot=plt.imshow(batch[0])
#     i+=1
#     if i%10 == 0:
#         break
plt.show()