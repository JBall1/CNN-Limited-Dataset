from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

import pandas as pd
import numpy as np

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.1, 0.9]
)


kernel_size = (3,3)
pool_size = (2,2)
first_filters = 32
second_filters = 64
third_filtesr = 128

dropout_conv = 0.3
dropout_dense = 0.3
IMAGE_SIZE = 150

CLASSES = 3

#Model setup
model = Sequential()
model.add(Conv2D(first_filters,kernel_size,activation='relu',input_shape = (IMAGE_SIZE,IMAGE_SIZE,3))) #This 3, would be a 1 if the images are gray
model.add(Conv2D(first_filters,kernel_size,activation='relu'))
model.add(Conv2D(first_filters,kernel_size,activation='relu'))
model.add(MaxPool2D(pool_size=pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(second_filters,kernel_size,activation='relu'))
model.add(Conv2D(second_filters,kernel_size,activation='relu'))
model.add(Conv2D(second_filters,kernel_size,activation='relu'))
model.add(MaxPool2D(pool_size=pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(third_filtesr,kernel_size,activation='relu'))
model.add(Conv2D(third_filtesr,kernel_size,activation='relu'))
model.add(Conv2D(third_filtesr,kernel_size,activation='relu'))
model.add(MaxPool2D(pool_size=pool_size))
model.add(Dropout(dropout_conv))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(dropout_dense))
model.add(Dense(3, activation='softmax'))

model.compile(Adam(lr=0.0001), loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

batch_size = 16

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    'Images',
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = train_datagen.flow_from_directory(
    'Images',
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=batch_size,
    class_mode='categorical'
)


filepath = 'tutorial.h5'
checkpoint = ModelCheckpoint(filepath,monitor='val_accuracy',verbose=1, save_best_only=True,mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',factor=0.5, patience=3, verbose=1, mode='max',min_lr=0.00001)
callsback = [checkpoint, reduce_lr]

fitting = model.fit(train_generator, steps_per_epoch=200//batch_size, validation_data=validation_generator, validation_steps=800//batch_size, epochs=30, verbose=1)