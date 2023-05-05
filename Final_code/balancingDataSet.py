import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from keras.preprocessing.image import ImageDataGenerator

y = []
label = 0
folders = sorted(os.listdir("Merged_Data/"))
for folder_path in folders:
    if folder_path == "Readme.txt":
        continue
    for file_path in os.listdir("Merged_Data/" + folder_path):
        if os.path.splitext(file_path)[-1] == '.ppm':
            y.append(label)
    label += 1

y = np.array(y)


plt.hist(y, bins=len(np.unique(y)))
plt.title("Distribution of data in dataset")
plt.xlabel("ClassId")
plt.ylabel("Frequency")
plt.show()

maxi = 0
for i in range(62):
    maxi = max(maxi, np.count_nonzero(y == i))

image_size=32
label = 0
folders = sorted(os.listdir("Merged_Data/"))
for folder_path in folders:
    if folder_path == "Readme.txt":
        continue
    X = []
    y = []
    for file_path in os.listdir("Merged_Data/" + folder_path):
        if os.path.splitext(file_path)[-1] == '.ppm':
            image = cv2.imread("Merged_Data/" + folder_path + "/" + file_path)
            X.append(cv2.resize(image, (image_size, image_size)))
            y.append(label)

    nb_images = len(X)

    X_gen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.5, 1.5],
        zoom_range=0.2
        #shear_range=10.0
    )

    X = np.array(X)
    y = np.array(y)

    X_gen = X_gen.flow(X, y, batch_size=1)
    X = list(X)
    for i in range(maxi-nb_images):
        image = X_gen.next()
        X.append(image[0][0].astype("uint8"))

    X = np.array(X)
    
    for i in range(len(X)):
        plt.imsave("Merged_and_Balanced_DataSet/" + folder_path + "/" + str(i) + ".ppm", cv2.cvtColor(X[i], cv2.COLOR_BGR2RGB))
    
    label += 1
