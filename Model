import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt


#Insert the pathway to your animal images files here
path = ''
images = np.load(os.path.join(path,'animalimages.npy'))
labels = np.load(os.path.join(path, 'animallabels.npy'))

images = images / 255.0

class_names = {}
i = 0
for folder in os.listdir(path):
    class_names.update({folder: i})
    i+=1

newlabels = []
for x in range(len(labels)):
    arr = []
    for y in range(len(class_names)):
        arr.append(0)
    arr[class_names[labels[x]]] = 1
    newlabels.append(arr)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(80, 80, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(155, activation='relu'))


model.summary()

model.compile(optimizer='adam',
 loss="categorical_crossentropy",
 metrics=['accuracy'])

history = model.fit(np.array(images), np.array(newlabels), epochs=1, batch_size=32, validation_split=0.2)



