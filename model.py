from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import time
import pandas as pd
import numpy as np

NAME = "Wine-Classification-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir= 'logs/{}'.format(NAME))

datagen = ImageDataGenerator(

    rescale = 1./255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest',
    brightness_range = [0.1, 0.9]
)

kernel_size = (3,3)
pool_size = (2,2)
first_filters = 32
second_filters = 64
third_filters = 128

dropout_conv = 0.3
dropout_dense = 0.3
IMAGE_SIZE = 150

CLASSES = 5

#Model setup
model = Sequential()
model.add(Conv2D(first_filters,kernel_size,activation='relu',input_shape = (IMAGE_SIZE,IMAGE_SIZE,3))) #This 3, would be a 1 if the images are gray
model.add(Conv2D(first_filters,kernel_size,activation='relu'))
model.add(Conv2D(first_filters,kernel_size,activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(second_filters,kernel_size,activation='relu'))
model.add(Conv2D(second_filters,kernel_size,activation='relu'))
model.add(Conv2D(second_filters,kernel_size,activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(dropout_conv))

# model.add(Conv2D(third_filters,kernel_size,activation='relu'))
# model.add(Conv2D(third_filters,kernel_size,activation='relu'))
# model.add(Conv2D(third_filters,kernel_size,activation='relu'))
# model.add(MaxPooling2D(pool_size=pool_size))
# model.add(Dropout(dropout_conv))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(dropout_dense))
model.add(Dense(4, activation='softmax'))

model.compile(Adam(lr=0.0001), loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

batch_size = 16

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    'Dataset Test',
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = train_datagen.flow_from_directory(
    'Dataset Test',
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=batch_size,
    class_mode='categorical'
)

# filepath = 'model.h5'
# checkpoint = ModelCheckpoint(monitor='val_accuracy',verbose=1, save_best_only=True,mode='max', filepath=filepath)

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',factor=0.5, patience=3, verbose=1, mode='max',min_lr=0.00005)
# callsback = [checkpoint, reduce_lr]

fitting = model.fit(train_generator, validation_data=validation_generator, epochs= 30, verbose=1)

# print(callsback)
print(train_generator.class_indices)
predictions = model.predict(train_generator, validation_generator)
predicted_classes = np.argmax(predictions, axis=1)
print(predictions)

true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())

report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

# model.save("model.h5")
# print(confusion_matrix(validation_generator.classes, predictions))

# print('Classification Report')
# target_names = ['Bellini Chianti Red Wine', 'Chateau Mukhrani Dessert Wine', 'Cotes De Gascogne White Wine', 'Fontana Candida Crascati White Wine']
# print(classification_report(validation_generator.classes, predictions, target_names=target_names))

# labels = '\n'.join(sorted(train_generator.class_indices.keys()))

# with open('labels-new.txt', 'w') as f:
#     f.write(labels)

# saved_model_dir = ''
# tf.saved_model.save(model, saved_model_dir)
# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
# tflite_model = converter.convert()

# with open('model-new.tflite', 'wb') as f:
#     f.write(tflite_model)