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

from keras.callbacks import ModelCheckpoint

SEED_VALUE = 42
 
# On fixed la seed pour rendre le code déterministe
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

# On parcourt le dataset augmenté et stocke les labels
image_size=32
X = []
y = []
count = 0
label = 0
folders = sorted(os.listdir("AugmentBalancedDataSetMerged/"))
for folder_path in folders:
    if folder_path == "Readme.txt":
        continue
    for file_path in os.listdir("AugmentBalancedDataSetMerged/" + folder_path):
        if os.path.splitext(file_path)[-1] == '.ppm' or os.path.splitext(file_path)[-1] == '.png':
            image = cv2.imread("AugmentBalancedDataSetMerged/" + folder_path + "/" + file_path)
            X.append(cv2.resize(image, (image_size, image_size)))
            count += 1
            y.append(label)
    label += 1

X = np.array(X)
y = np.array(y)

# On permute les données pour ne pas donner des entrées triées au CNN 
p = np.random.permutation(len(X))

X_train = X[p]
y_train = y[p]
    
# On normalise les images pour avoir des valeurs entre 0 et 1
X_train = X_train.astype("float32") / 255
 
# On convertit les labels en one-hot encoding
y_train = to_categorical(y_train)
 

# Modèle pris depuis le site https://learnopencv.com/Implementing-cnn-tensorflow-keras/ très similaire à l'architecture du VGG-16
def cnn_model_dropout(input_shape=(32, 32, 3)):
     
    model = Sequential()
     
    #------------------------------------
    # Premier bloc de convolution
    #------------------------------------
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    #------------------------------------
    # Deuxième bloc de convolution
    #------------------------------------   
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    #------------------------------------
    # Troisième bloc de convolution
    #------------------------------------
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    #------------------------------------
    # Couche entièrement connectée
    #------------------------------------
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(label, activation='softmax'))
     
    return model

model = cnn_model_dropout()

model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'],
            )

# On sauvegarde les paramètres du modèle après chaque epoch pour pouvoir les tester après
filepath = "my_balanced_model_merged_RMS_32_{epoch:02d}_Augmented.h5"

checkpoint = ModelCheckpoint(filepath, save_freq='epoch', save_weights_only=False, period=1)

# On entraine le modèle et garde l'historique pour pouvoir plot après
history = model.fit(X_train,
                    y_train,
                    batch_size=32, 
                    epochs=15, 
                    verbose=1, 
                    validation_split=.3,
                    callbacks=[checkpoint]
                   )

# Fonction également prise du site https://learnopencv.com/Implementing-cnn-tensorflow-keras/ mais très facilement faisable par nous-même
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
    plt.xlim([0, 15-1])
    plt.ylim(ylim)
    # On met des ticks sur l'axe des x
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    plt.grid(True)
    plt.legend(metric_name)
    plt.savefig(f"CNN_plot_{ylabel}.eps")   
    plt.show()
    plt.close()

# On plot les résultats
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

