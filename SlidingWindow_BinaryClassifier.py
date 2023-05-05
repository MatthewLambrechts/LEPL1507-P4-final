""" 
    Sliding Window + Background&PanelModel

    Stratégie:
    3 étapes:
    - sliding window
    - BinaryModel sur chaque sous image pour éliminer la large majorité d'images de background.
    - Background&PanelModel sur chaque sous-image restante et stockage ou non dans le dictionnaire des meilleures prédictions de chaque sous-image
    - détermination de la meilleure prédiction parmi les meilleures prédictions de chaque sous-image (vérification avec CroppedModel)


 """

import cv2
import time 
from keras import models
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os


# Yields different sub_image sizes (but not the images themselves)
def pyramid_2(image, scale=1.5, minSize=(16, 16)):
 	# yield the original image size (only non-square sub_image returned)
    yield (image.shape[0],image.shape[1])
    it = 0
    #get minimum of height and width
    m = min(image.shape[0]*scale, image.shape[1]*scale)
    while it < 8:
  		  # compute the new dimensions of the image and resize it
          m = int(m/scale)
          it += 1
  		  # if the resized image does not meet the supplied minimum
  		  # size, then stop constructing the pyramid
          # If an image 3 times bigger has already be considered as non background,
          # stop constructing the pyramid
          if STOP_MARKER == 3 or m < minSize[0] or m < minSize[1]:
              break
  		  # yield the next size in the pyramid (square images)
          yield (m,m)

# Yields all sub_images of given size 
def sliding_window_2(image, windowSize):
    #Slide de window_frame across the whole image
    for y in range(0, image.shape[0], int(windowSize[0]/2.0)):
          for x in range(0, image.shape[1], int(windowSize[1]/2.0)):
              #If the border of the image is reached, stop
              if(y + windowSize[0]/2 > image.shape[0] or x + windowSize[1]/2 > image.shape[1]) : break
              # yield the current window
              yield (x, y, image[y:y + windowSize[0], x:x + windowSize[1]])


#dictionnaire des labels: {num_panneau: nom_panneau, ...}
dico = {}
df = pd.read_excel("LEPL1507_TS.xlsx")
for index, row in df.iterrows():
    dico[row["c1"]] = row["c2"]
dico[62] = "Background"


#Load all three different models
binaryModel = models.load_model("my_background&pannel_binary_model")
pannelModel = models.load_model("my_background&panel_model")
cropped_model = models.load_model("my_balanced_model_merged_RMS_32_15_Augmented.h5")


#Get all images and labels
image_size=32
X = []
Y = []
count = 0
folders_path = "challenge-1/eval_kaggle1_sorted"
folders_path = "challenge-2-non-cropped-images/eval_kaggle2_sorted"
folders = sorted(os.listdir(folders_path))
for folder_path in folders:
    label = int(folder_path)
    images = sorted(os.listdir(folders_path + "/" + folder_path), key=lambda f: int(f.split(".ppm")[0]))
    for image_path in images :
        image = cv2.imread(folders_path + "/" + folder_path + "/" + image_path)
        X.append(image)
        count += 1
        Y.append(label)


IM_index = 0
NBR_correct = 0
t = time.time()
#Predict class for each image
for image in X :
    
    #Show image
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.show()
    
    print(f"Image {IM_index + 1}")
    
    probas_predictions = [] # To store probabilities of kept images
    class_predictions = []  # To store class predictions of kept images
    kept_images = []  # To store kept images
    threshold = 0.99  # Threshold for binary classifier
    threshold2 = 0.95 # Threshold for pannel + background model
    
    
    
    # loop over the image pyramid
  
    STOP_MARKER = 0
    
    # Loop over different sub_image sizes
    for size in pyramid_2(image, scale=1.5):
        
        # Whether an image with this size has been considered as a pannel - useful later
        Size_contains_non_BK = False
        
        # loop over the sliding window for each size given by pyramid
        for (x, y, resized) in sliding_window_2(image, windowSize=size):
            
            # Some comments are used to plot the sliding window/results - not essential to compute result
            original_image_clone = image.copy()
            
            # Resized is the current window/cut image
            # We copy it to avoid modifying original image
            clone = resized.copy()
            
            # Resize image to pass it to our models
            img = cv2.resize(clone, (32, 32))
            img = img.astype("float32") / 255
            
            
            # Get probability of window being background
            model_predictions = binaryModel.predict(np.expand_dims(img, axis=0), verbose=0)
    
            # Sort the results - tell if probability for background > probability for pannel
            # If idx[0] = 1 -> proba background > proba pannel
            # If idx[0] = 0 -> proba background < proba pannel
            idx = np.argsort(model_predictions.flatten())[::-1]
    
            #If image isn't a background (use threshold to determine) : 
            if(idx[0] != 1 and model_predictions[0][idx[0]] > threshold):
                
                # Get predictions of TS + background model (background class == 62)
                pannel_predictions = pannelModel.predict(np.expand_dims(img, axis=0), verbose=0)
                idx2 = np.argsort(pannel_predictions.flatten())[::-1][:3]
                
                #If image is still not classified as background (use second threshold to determine) :
                if idx2[0] != 62 and pannel_predictions[0][idx2[0]] > threshold2 :
                    
                    # cv2.rectangle(original_image_clone, (x, y), (x + size[0], y + size[1]), (0, 0, 255), 5)
                    # plt.imshow(cv2.cvtColor(original_image_clone, cv2.COLOR_BGR2RGB))
                    # plt.show()
                    
                    # Save the results for this image
                    probas_predictions.append(pannel_predictions[0][idx2[0]])
                    class_predictions.append(idx2[0])
                    kept_images.append(img)
                    
                    # If this is the first image of this size to be considered as a pannel
                    if Size_contains_non_BK == False :
                        STOP_MARKER += 1 #When this reaches 3 (or 2 - TO MODIFY), an image 3 times bigger has been registered as a pannel : stop the sliding window (small images sometimes cause errors)
                        Size_contains_non_BK = True
        
            
            # else :
            #     cv2.rectangle(original_image_clone, (x, y), (x + size[0], y + size[1]), (0, 255, 0), 5)
            #     cv2.imshow("Window", original_image_clone)
            #     cv2.waitKey(1)
 
        # If the no image of this size has been registered, restore the size marker
        # (often, images around the pannel but a bit too big for it, as well as images on the pannel but a bit too small are registered as pannel)
        if Size_contains_non_BK == False : STOP_MARKER = 0
    
    # cv2.destroyAllWindows()
    
    
    
    #########################################################################################################################################################################
    
    # probas_predictions now contains the probability for each kept sub_image
    # If empty, no sub_image was kept
    if probas_predictions == [] :
        print("Found no pannel : returning random answer")
        best_prediction = np.random.randint(62)
        print("Best prediction: classID = ", best_prediction, ", name = ", dico[best_prediction])
    
    else :
        found_same = False
        sames_proba = []
        sames_class = []
        
        #Compare the result af pannel + background model to that of the cropped model
        for i in range(len(probas_predictions)):
            cropped_model_predictions = cropped_model.predict(np.expand_dims(kept_images[i], axis=0), verbose = 0)
            idx3 = np.argsort(cropped_model_predictions.flatten())[::-1][:3]
            if(idx3[0] == class_predictions[i]) :
                found_same = True
                
                # Ponderate the probas by multiplying results of both models
                sames_proba.append(cropped_model_predictions[0][idx3[0]]*probas_predictions[i])
                sames_class.append(idx3[0])
        
        # If both models return same result for a sub_image
        if found_same :
            print(sames_proba)
            print(sames_class)
            found_very_good_big_image = False
            
            # Try first the big images - sometimes very small images produce strange good results
            # When a pannel is realy recognised, proba is often > 0.999. Also only prefer a bigger image if absolutely certain of the result.
            for i in range(len(sames_proba)) :
                if sames_proba[i] > 0.999 :
                    print("Found very good big image")
                    found_very_good_big_image = True
                    best_prediction = sames_class[i]
                    print("Best prediction: classID=", best_prediction, ", name=", dico[best_prediction])
                    break
            if found_very_good_big_image == False :
                print("Found no very good big image")
                index = np.argmax(sames_proba)
                best_prediction = sames_class[index]
                print("Best prediction: classID=", best_prediction, ", name=", dico[best_prediction])
        
        # If no sub_image has same results on both models
        else :
            print("Found no two same predictions")
            index = np.argmax(probas_predictions)
            best_prediction = class_predictions[index]
            print("Best prediction: classID=", best_prediction, ", name=", dico[best_prediction])
    
    print(f"Correct answer : {Y[IM_index]}, {dico[Y[IM_index]]}")
    if(best_prediction == Y[IM_index]) : NBR_correct += 1
    IM_index += 1
        
t = time.time() - t
# Compute accuracy
Accuracy = NBR_correct/IM_index

print(f"Accuracy = {Accuracy}")
print(f"Time Taken : {t}, Per Image : {t/66}")
