from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import PIL
import PIL.Image
import os
import time

# Dimensions of our images
img_width, img_height = 580, 180

# Image folder
folder_path = 'test/'

# Path to model
model_path = 'ecg_test_model.h5'

# load the trained model
model = load_model(model_path)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# load all images into a list
for img in os.listdir(folder_path):
    start_time = time.time()
    img = os.path.join(folder_path, img)
    print(img)
    img = image.load_img(img, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    y_prob = model.predict(images, batch_size=10)
    print(y_prob)
    y_classes = y_prob.argmax(axis=-1)
    print(y_classes) #0 = abnormal, 1 = normal
    inference_time = (time.time() - start_time)*1000
    print("Inference time: %s milliseconds" % inference_time)
    time.sleep(1)
