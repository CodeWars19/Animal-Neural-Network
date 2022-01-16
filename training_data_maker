import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

training_images = []
training_labels = []'

#Set this path to the path where you saved your simple_images folder
path = ''
for folder in os.listdir(path):
    newpath = path + folder
    try:
        for img in os.listdir(newpath):
            try:
                pic = cv2.imread(os.path.join(newpath, img))
                pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
                pic = cv2.resize(pic, (80, 80))
                training_images.append(pic)
                training_labels.append(folder)
            except cv2.error:
                pass
    except NotADirectoryError:
        pass

#Set this path to where you want to save the numpy arrays of the images and labels of the animals
newpath = ''
np.save(os.path.join(path,'animalimages'), np.array(training_images))
np.save(os.path.join(path,'animallabels'), np.array(training_labels))
