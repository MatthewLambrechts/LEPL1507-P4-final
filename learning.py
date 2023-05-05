import os
import random
import numpy as np
import matplotlib.pyplot as plt

import cv2

import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
 
from keras.utils import to_categorical
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)

SEED_VALUE = 42
 
# Fix seed to make training deterministic.
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

image_size=32
X = np.empty([4575, image_size, image_size, 3], dtype='uint8')
y = np.empty(0)
count = 0
label = 0
folders = sorted(os.listdir("../LEPL1507/LEPL1507-Projet4/Training"))
for folder_path in folders:
    if folder_path == "Readme.txt":
        continue
    for file_path in os.listdir("../LEPL1507/LEPL1507-Projet4/Training/" + folder_path):
        if os.path.splitext(file_path)[-1] == '.ppm':
            image = cv2.imread("../LEPL1507/LEPL1507-Projet4/Training/" + folder_path + "/" + file_path)
            X[count, :, :, :] = cv2.resize(image, (image_size, image_size))
            count += 1
            y = np.append(y,label)
    label += 1

p = np.random.permutation(len(X))

X_train = X[p]
y_train = y[p]

# (X_train, y_train), (X_test, y_test) = cifar10.load_data()
 
# print(X_train.shape)
# print(X_test.shape)

# plt.figure(figsize=(18, 9))
 
# num_rows = 4
# num_cols = 8
 
# # plot each of the images in the batch and the associated ground truth labels.
# for i in range(num_rows*num_cols):
#     ax = plt.subplot(num_rows, num_cols, i + 1)
#     plt.imshow(X_train[i,:,:])
#     plt.axis("off")

# plt.show()

# Normalize images to the range [0, 1].
X_train = X_train.astype("float32") / 255
#X_test  = X_test.astype("float32") / 255
 
# Change the labels from integer to categorical data.
print('Original (integer) label for the first training sample: ', y_train[0])
 
# Convert labels to one-hot encoding.
y_train = to_categorical(y_train)
#y_test  = to_categorical(y_test)
 
print('After conversion to categorical one-hot encoded labels: ', y_train[0])

def cnn_model_dropout(input_shape=(32, 32, 3)):
     
    model = Sequential()
     
    #------------------------------------
    # Conv Block 1: 32 Filters, MaxPool.
    #------------------------------------
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    #------------------------------------
    # Conv Block 2: 64 Filters, MaxPool.
    #------------------------------------
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    #------------------------------------
    # Conv Block 3: 64 Filters, MaxPool.
    #------------------------------------
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
     
    #------------------------------------
    # Flatten the convolutional features.
    #------------------------------------
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(label, activation='softmax'))
     
    return model

# Create the model.
model = cnn_model_dropout()
model.summary()

model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'],
             )

history = model.fit(X_train,
                    y_train,
                    batch_size=256, 
                    epochs=31, 
                    verbose=1, 
                    validation_split=.3,
                   )

def plot_results(metrics, title=None, ylabel=None, ylim=None, metric_name=None, color=None):
     
    fig, ax = plt.subplots(figsize=(15, 4))
 
    if not (isinstance(metric_name, list) or isinstance(metric_name, tuple)):
        metrics = [metrics,]
        metric_name = [metric_name,]
         
    for idx, metric in enumerate(metrics):    
        ax.plot(metric, color=color[idx])
     
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim([0, 31-1])
    plt.ylim(ylim)
    # Tailor x-axis tick marks
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    plt.grid(True)
    plt.legend(metric_name)   
    plt.show()
    plt.close()

# Retrieve training results.
train_loss = history.history["loss"]
train_acc  = history.history["accuracy"]
valid_loss = history.history["val_loss"]
valid_acc  = history.history["val_accuracy"]
   
plot_results([ train_loss, valid_loss ],        
            ylabel="Loss", 
            ylim = [0.0, 5.0],
            metric_name=["Training Loss", "Validation Loss"],
            color=["g", "b"]);
 
plot_results([ train_acc, valid_acc ], 
            ylabel="Accuracy",
            ylim = [0.0, 1.0],
            metric_name=["Training Accuracy", "Validation Accuracy"],
            color=["g", "b"])

	
# Using the save() method, the model will be saved to the file system in the 'SavedModel' format.
model.save('my_model_parking')

