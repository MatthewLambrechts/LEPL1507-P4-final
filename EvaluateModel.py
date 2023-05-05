# -*- coding: utf-8 -*-
"""
Created on Mon May  1 11:19:23 2023

@author: Valentin
"""

from keras import models
import os 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import to_categorical

import pandas as pd 

dico = {}
df = pd.read_excel("LEPL1507_TS.xlsx")
for index, row in df.iterrows():
    dico[row["c1"]] = row["c2"]

model = models.load_model('my_balanced_model_merged_RMS_32_15_Augmented.h5')



image_size=32
X = []
Y = []
count = 0
folders_path = "challenge-1/eval_kaggle1_sorted"
# folders_path = "challenge-2-non-cropped-images/eval_kaggle2_sorted"
folders_path = "Testing"
folders = sorted(os.listdir(folders_path))
for folder_path in folders:
    label = int(folder_path)
    images = sorted(os.listdir(folders_path + "/" + folder_path))
    for image_path in images :
        if(image_path.endswith(".ppm")) :
            image = cv2.imread(folders_path + "/" + folder_path + "/" + image_path)
            X.append(cv2.resize(image, (image_size, image_size)))
            count += 1
            Y.append(label)

X = np.array(X)
Y = np.array(Y)

##########################################################################################
# Model Selection :
# my_model_RMS_128_30_DA: Testing: 95.67%  19: 88.52%  21: 92.31%  22: 59.9%   37: 90.28%
# my_model_RMS_32_31_DA : Testing: 96.83%  19: 91.9%   21: 95.64%  22: 83.87%  37: 75.83%
# my_model_RMS_32_15_DA : Testing: 94%     19: 93.29%  21: 96%     22: 96.31%  37: 71.38%
# my_model+_RMS_32_15_DA: Testing: 95.87%  19: 94.21%  21: 90.64%  22: 98.38%  37: 70.83%
##########################################################################################



# Normalize images to the range [0, 1].
X = X.astype("float32") / 255

# Convert labels to one-hot encoding.

Y = to_categorical(Y)

def evaluate_model(dataset, model, labels):
 
    # class_names = ['airplane',
    #                'automobile',
    #                'bird',
    #                'cat',
    #                'deer',
    #                'dog',
    #                'frog',
    #                'horse',
    #                'ship',
    #                'truck' ]
     
    # Retrieve a number of images from the dataset.
    data_batch = dataset
 
    # Get predictions from model.  
    predictions = model.predict(data_batch)
 
    #plt.figure(figsize=(20, 8))
    num_matches = 0
         
    for idx in range(len(data_batch)):
        # ax = plt.subplot(num_rows, num_cols, idx + 1)
        # plt.axis("off")
        # plt.imshow(data_batch[idx])
 
        pred_idx = np.argmax(predictions[idx])
        #category.append(pred_idx)
        truth_idx = np.nonzero(labels[idx])
             
        # title = str(class_names[truth_idx[0][0]]) + " : " + str(class_names[pred_idx])
        # title = "test"
        # title_obj = plt.title(title, fontdict={'fontsize':13})
             
        if pred_idx == truth_idx:
            num_matches += 1    
        #     plt.setp(title_obj, color='g')
        # else:
        #     plt.setp(title_obj, color='r')
                 
    acc = num_matches/len(data_batch)
    print("Prediction accuracy: ", acc)
    return

evaluate_model(X, model, Y)