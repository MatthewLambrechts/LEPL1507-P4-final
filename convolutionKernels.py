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
model2 = models.load_model('my_model_parking')

image_size=32
X = []
y = []
count = 0
label = 45
folders = sorted(os.listdir("Testing_parking"), key=lambda f: int(f.split(".ppm")[0]))
for folder_path in folders:
    images = sorted(os.listdir("Testing_parking/"+ folder_path), key=lambda f: int(f.split(".ppm")[0]))
    for image_path in images :
        image = cv2.imread("Testing_parking/" + folder_path + "/" + image_path)
        X.append(image)
        count += 1
        y.append(label)
    label += 1

X = np.array(X)
y = np.array(y)

##########################################################################################
# Model Selection :
# my_model_RMS_128_30_DA: Testing: 95.67%  19: 88.52%  21: 92.31%  22: 59.9%   37: 90.28%
# my_model_RMS_32_31_DA : Testing: 96.83%  19: 91.9%   21: 95.64%  22: 83.87%  37: 75.83%
# my_model_RMS_32_15_DA : Testing: 94%     19: 93.29%  21: 96%     22: 96.31%  37: 71.38%
# my_model+_RMS_32_15_DA: Testing: 95.87%  19: 94.21%  21: 90.64%  22: 98.38%  37: 70.83%
##########################################################################################

# image_size=32
# X = []
# y = []
# count = 0
# label = 0
# folders = sorted(os.listdir("Testing"))
# for folder_path in folders:
#     if folder_path == "Readme.txt":
#         continue
#     for file_path in os.listdir("Testing/" + folder_path):
#         if os.path.splitext(file_path)[-1] == '.ppm':
#             image = cv2.imread("Testing/" + folder_path + "/" + file_path)
#             X.append(cv2.resize(image, (image_size, image_size)))
#             count += 1
#             y.append(label)
#     label += 1

# X = np.array(X)
# y = np.array(y)

# image_size=32
# X = []
# y = []
# count = 0
# label = 37
# name = "Testing" + str(label)
# folders = sorted(os.listdir(name))
# for file_path in folders:
#     if os.path.splitext(file_path)[-1] == '.png':
#         image = cv2.imread(name + "/" + file_path)
#         X.append(cv2.resize(image, (image_size, image_size)))
#         count += 1
#         y.append(label)

# X = np.array(X)
# y = np.array(y)

# Normalize images to the range [0, 1].
#X = X.astype("float32") / 255

# Convert labels to one-hot encoding.
y = to_categorical(y)
print(y.shape)

def evaluate_model(dataset, model):
 
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
        truth_idx = np.nonzero(y[idx])
             
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

#evaluate_model(X, model)

# cm = tf.math.confusion_matrix(labels=tf.argmax(y, axis=1), predictions=category)
 
# # Plot the confusion matrix as a heatmap.
# plt.figure(figsize=[14, 7])
# import seaborn as sn
# sn.heatmap(cm, annot=True, fmt='d', annot_kws={"size": 12})
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Truth')
# plt.show()

# df["Category"] = category
# df.to_csv("my_balanced_model_merged_RMS_32_15_Augmented.csv", index=False)

# i = 420

# plt.imshow(X[i])
# plt.show()

# print("Label : ", np.argmax(model.predict(np.expand_dims(X[i], axis=0))))

# i = 100
# predictions = model.predict(X)
# print(predictions[i])
# print(np.argmax(predictions[i]), dico[np.argmax(predictions[i])])
# plt.imshow(X[i])
# plt.show()
# print("real = ", np.argmax(y[i]), dico[np.argmax(y[i])])

def isbright(image, dim=5, thresh=0.5):
    # Resize image to 10x10
    image = cv2.resize(image, (dim, dim))
    # Convert color space to LAB format and extract L channel
    L, A, B = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    # Normalize L channel by dividing all pixel values with maximum pixel value
    L = L/np.max(L)
    # Return True if mean is greater than thresh else False
    return np.mean(L) > thresh

# pos = 0

# for file in sorted(os.listdir("challenge-1/eval_kaggle1"), key=lambda f: int(f.split(".ppm")[0])):
#     if os.path.splitext(file)[-1] == '.ppm' or os.path.splitext(file)[-1] == '.png' or os.path.splitext(file)[-1] == '.jpeg' or os.path.splitext(file)[-1] == '.jpg' or os.path.splitext(file)[-1] == '.webp':
#         test = cv2.imread("challenge-1/eval_kaggle1/" + file)
#         plt.imshow(cv2.cvtColor(test, cv2.COLOR_BGR2RGB))
#         plt.show()
#         test = cv2.resize(test, (32, 32))
#         test = test.astype("float32") / 255
#         #idx = np.argmax(model.predict(np.expand_dims(test, axis=0)))
#         model_predictions = model.predict(np.expand_dims(test, axis=0))
#         idx = np.argsort(model_predictions.flatten())[::-1][:3]
#         print(idx[0], dico[idx[0]], ", correct answer:", np.nonzero(y[pos])[0][0])
#         print(idx[0], ":", model_predictions[0][idx[0]], ",", idx[1], ":", model_predictions[0][idx[1]], ",", idx[2], ":", model_predictions[0][idx[2]]) 
#         #print(isbright(test))
#         pos += 1

# test = cv2.imread("Capture d’écran du 2023-02-14 09-24-51.png")
# test = cv2.resize(test, (32, 32))
# test = test.astype("float32") / 255
# idx = np.argmax(model.predict(np.expand_dims(test, axis=0)))
# print(idx, dico[idx])

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def run_histogram_equalization(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    #configure CLAHE
    clahe = cv2.createCLAHE(clipLimit=10,tileGridSize=(5, 5))

    #0 to 'L' channel, 1 to 'a' channel, and 2 to 'b' channel
    img[:,:,0] = clahe.apply(img[:,:,0])

    img = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)
    return img



kernel1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
kernel2 = 1/256.0 * np.array([[1,4,6,4,1], [4,16,24,16,4], [6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]])
kernel3 = -1/256.0 * np.array([[1,4,6,4,1], [4,16,24,16,4], [6,24,-476,24,6],[4,16,24,16,4],[1,4,6,4,1]])
kernel4 = 1/16.0 * np.array([[1,2,1], [2,4,2],[1,2,1]])
kernel5 = 1/9.0 * np.array([[1,1,1],[1,1,1],[1,1,1]])
kernel6 = np.array([[0,-1,0], [-1,5,-1],[0,-1,0]])
kernels = [kernel1, kernel2, kernel3, kernel4, kernel5, kernel6]
category = []
pos = 1
correct = 0
for i in range(0,len(X),15):
    # image_base = cv2.imread(f'Testing_parking/{pos}'+j+'.ppm')
    # plt.imshow(cv2.cvtColor(image_base, cv2.COLOR_BGR2RGB))
    # plt.show()
    image_base = X[i]
    image = cv2.resize(image_base, (32, 32))
    image = image.astype("float32") / 255
    if np.max(model.predict(np.expand_dims(image, axis=0), verbose=0)) < 0.80:
        #image = increase_brightness(image, 100)
        image = cv2.convertScaleAbs(image_base, alpha=1, beta=20)
        #image = run_histogram_equalization(image_base)
        image = cv2.filter2D(image, -1, kernel1)
        image = cv2.resize(image, (32, 32))
        image = image.astype("float32") / 255
    model_predictions = model.predict(np.expand_dims(image, axis=0), verbose=0)
    idx = np.argsort(model_predictions.flatten())[::-1][:3]
    print("Image", i+1)
    #print(idx[0], dico[idx[0]], ", correct answer:", np.nonzero(y[i-1])[0][0])
    #print(idx[0], ":", model_predictions[0][idx[0]], ",", idx[1], ":", model_predictions[0][idx[1]], ",", idx[2], ":", model_predictions[0][idx[2]]) 
    
    if idx[0] in [45,46,47,48,49,50]  and np.max(model_predictions) < 0.99:
        print(f"Original prediction : {idx[0]}")
        print(f"Original certainty : {model_predictions[0][idx[0]]}")
        plt.imshow(cv2.cvtColor(image_base, cv2.COLOR_BGR2RGB))
        plt.title(f"{idx[0]}")
        plt.show()
        image = cv2.resize(image_base, (32, 32))
        image = image.astype("float32") / 255
        model_predictions = model2.predict(np.expand_dims(image, axis=0), verbose=0)
        if np.max(model_predictions) < 0.98:
            #image = increase_brightness(image, 100)
            image = cv2.convertScaleAbs(image_base, alpha=1, beta=20)
            #image = run_histogram_equalization(image_base)
            image = cv2.filter2D(image, -1, kernel1)
            image = cv2.resize(image, (32, 32))
            image = image.astype("float32") / 255
            model_predictions = model2.predict(np.expand_dims(image, axis=0), verbose=0)
        idx = np.argsort(model_predictions.flatten())[::-1][:3] + 45
        print(idx[0], dico[idx[0]], ", correct answer:", np.nonzero(y[i-1])[0][0], dico[np.nonzero(y[i-1])[0][0]])
        print(idx[0], ":", model_predictions[0][idx[0]-45], ",", idx[1], ":", model_predictions[0][idx[1]-45], ",", idx[2], ":", model_predictions[0][idx[2]-45])
    
    if (idx[0] == np.nonzero(y[i-1])[0][0]):
        correct += 1
    category.append(idx[0])
    pos += 1

print(correct/(len(X)//15))

# df = pd.DataFrame()
# df["Id"] = np.arange(1, len(X)+1)
# df["Category"] = category
# df.to_csv("my_balanced_model_merged_RMS_32_15_Augmented_Preprocess.csv", index=False)

# pos = 72

# image_base = cv2.imread(f'/home/roy/Documents/TestingLEPL1507/LEPL1507-Projet4/challenge-1/eval_kaggle1/{pos}.ppm')
# image = run_histogram_equalization(image_base)

# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.show()

# image = cv2.resize(image, (32, 32))
# image = image.astype("float32") / 255

# model_predictions = model.predict(np.expand_dims(image, axis=0), verbose=0)
# idx = np.argsort(model_predictions.flatten())[::-1][:3]
# print("Image", 1)
# print(idx[0], dico[idx[0]], ", correct answer:", np.nonzero(y[pos-1])[0][0])
# print(idx[0], ":", model_predictions[0][idx[0]], ",", idx[1], ":", model_predictions[0][idx[1]], ",", idx[2], ":", model_predictions[0][idx[2]]) 