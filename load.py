from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os

# dimensions of our images
img_width, img_height = 150, 150

# load the model we saved
model = load_model('model.h5')

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

# predicting images
img = image.load_img(r'C:\Users\Shahed\Desktop\test.jpg', target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict_classes(images, batch_size=10)
print (classes)

# predicting multiple images at once
img = image.load_img(r'Dataset Test\Fontana Candida Crascati White Wine\IMG_1586.jpg', target_size=(img_width, img_height))
y = image.img_to_array(img)
y = np.expand_dims(y, axis=0)

# pass the list of multiple images np.vstack()
images = np.vstack([x, y])
classes = model.predict_classes(images, batch_size=10)

# print the classes, the images belong to
print (classes)
# print (classes([0]))
# print (classes([0][0]))