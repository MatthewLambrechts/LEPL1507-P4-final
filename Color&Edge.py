import matplotlib.pyplot as plt
import cv2
import os 
import numpy as np
import pandas as pd
from keras import models


def extract_color(image, color):
    """Given an image and a color, return a filtered version of the image where only the specified color remains.
       Furthermore, returns the conditions under which a contour and associated rectangle are interesting to keep for the given color.
       For example, these conditions include imposing that the rectangle is square enough.
       Further still, returns the minimum size a rectangle should have for a contour of this color,
       and the percentage one should enlargen the obtained rectangle in order to enclose the whole traffic sign.
       Finally, returns the conditions under which a rectangle is considered 'special' and the conditions to merge two of these special rectangles.
       Please read the report for the full explanation of 'special' rectangles (under 'Modèle final')."""
    
    #Get image info and set default values
    height, width, c = image.shape
    invert = False
    two_masks = False
    special_conditions_1 = lambda w,h,minsize,cnt : False   #The first special conditions depends on the contour itself
    special_conditions_2 = lambda rec1, rec2 : False        #While the second depends on the relative size and position of the associated values
    special_conditions = [special_conditions_1,special_conditions_2]   #By default, no rectangle is considered 'special'
    
    # For each color, specify the bounds for the mask, as well as the associated conditions, minimum size and percentage
    if color == "blue" :
        lower_bound = np.array([100,100,50])
        upper_bound = np.array([140,255,255])
        conditions = lambda w,h,minsize,peri : w >= minsize and h >= minsize and 4/5 * w <= h <= 6/4 * w and peri < 0.27*w*h
        percentage = 0.1
        minsize = max(min(height,width)/30, 16)
        
    elif color == "red" :
        #Red spans around 180 / 0 in the HSV format, so we need two sets of bounds.
        two_masks = True
        lower_bound = np.array([0,100,0])
        upper_bound = np.array([20,255,255])
        lower_bound2 = np.array([160,100,0])
        upper_bound2 = np.array([180,255,255])
        
        conditions = lambda w,h,minsize,peri : w >= minsize and h >= minsize and 4/5 * w <= h <= 5/4 * w and peri < 0.27*w*h
        
        #Here special conditions apply, for the no_entry traffic sign
        special_conditions_1 = lambda w,h,minsize,peri : w >= minsize and h >= minsize/2.5 and 0.4 * w <= h <= 0.5 * w and peri < 0.27*w*h
        special_conditions_2 = lambda rec1, rec2 : abs(rec1[0]-rec2[0]) <= 5 and abs(rec1[2]-rec2[2]) <= 8 and abs(rec1[3]-rec2[3]) <= 6 and 3/4*rec1[3] <= abs(rec1[1]-rec2[1]) <= 2*rec1[3]
        special_conditions = [special_conditions_1,special_conditions_2]
        
        percentage = 0.1
        minsize = max(min(height,width)/30, 16)
        
    elif color == "white" :
        lower_bound = np.array([0,0,100])
        upper_bound = np.array([180,40,255])
        conditions = lambda w,h,minsize,peri : w >= minsize and h >= minsize and 2/3 * w <= h <= 3/2 * w  # and cnt.shape[0] < 0.15*w*h
        percentage = 0.5
        minsize = max(min(height,width)/35, 14)
        
    elif color == "black" :
        
        #Invert the final mask otherwise we get black pixels on a black background...
        invert = True
        lower_bound = np.array([0,50,0])
        upper_bound = np.array([255,255,60])
        conditions = lambda w,h,minsize,peri : w >= minsize and h >= minsize and 3/4 * w <= h <= 4/3 * w and peri < 0.27*w*h
        percentage = 0.1
        minsize = max(min(height,width)/25, 16)
        
    elif color == "orange" :
        
        lower_bound = np.array([10,50,0])
        upper_bound = np.array([30,255,255])
        conditions = lambda w,h,minsize,peri : w >= minsize and h >= minsize and 6/7 * w <= h <= 7/6 * w and peri < 0.15*w*h
        
        #Here as well, special conditions apply for the end_priority traffic sign
        special_conditions_1 = conditions
        special_conditions_2 = lambda rec1, rec2 : overlap_percentage(rec1, rec2) >= 0.35 and abs(rec1[2]-rec2[2]) <= 8 and abs(rec1[3]-rec2[3]) <= 8 and abs(rec1[0]-rec2[0]) >= rec1[2]/4 and abs(rec1[1]-rec2[1]) >= rec1[3]/4
        special_conditions = [special_conditions_1,special_conditions_2]
        percentage = 0.75
        minsize = max(min(height,width)/40, 10)
    
    #If the color is not specified or unexpected, return the non filtered original image, default conditions, percentage etc...
    else :
        conditions = lambda w,h,minsize,peri : w >= minsize and h >= minsize and 3/4 * w <= h <= 4/3 * w and peri < 0.27*w*h
        percentage = 0.1
        minsize = max(min(height,width)/25, 16)
        return image, conditions, special_conditions, percentage, minsize
    
    #Transform the image to HSV format
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Compute the pixels which are inside the bounds, in the form of a mask
    mask = cv2.inRange(frame, lower_bound, upper_bound)
    if two_masks :
        mask2 = cv2.inRange(frame, lower_bound2, upper_bound2)
        mask = mask + mask2
    
    #This is used for the black filter : convert first all pixels to a uniform color different then black before applying the mask
    if invert :
        frame = np.full_like(frame, 100)
    
    #Apply the mask and return the results
    result = cv2.bitwise_and(frame, frame, mask = mask)
    return  cv2.cvtColor(result, cv2.COLOR_HSV2BGR), conditions, special_conditions, percentage, minsize


#Given two rectangles, computes the overlap percentage as the ratio of the overlapping area wrt the area of the bigger rectangle
def overlap_percentage(rec1,rec2):
    x1,y1,w1,h1 = rec1
    x2,y2,w2,h2 = rec2
    x,y = max(x1,x2), max(y1,y2)
    w,h = max(min(x1+w1,x2+w2)-x,0), max(min(y1+h1,y2+h2)-y,0)
    perc = w*h/max(w1*h1,w2*h2)
    return perc

#Given two rectangles, create a new one englobing fully both rectangles
def merge_rectangles(rec1, rec2):
    x1,y1,w1,h1 = rec1
    x2,y2,w2,h2 = rec2
    x,y = min(x1,x2), min(y1,y2)
    w,h = max(x1+w1,x2+w2)-x , max(y1+h1,y2+h2)-y
    return (x,y,w,h)


#Adjust the gamma parameter of the image. If gamma > 0, brigthens the image, otherwise darken it.
#This is an OpenCV function, as fond at : https://pyimagesearch.com/2015/10/05/opencv-gamma-correction/
def adjust_gamma(image, gamma=1.0):
    new_image = np.zeros(image.shape, image.dtype)

    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    lookUpTable = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # Apply gamma correction using the lookup table
    new_image = cv2.LUT(image, lookUpTable)

    return new_image



def contours(original_image, conditions, special_conditions, percentage, minsize):
    """Find the contours in an image thanks to edge detection, then return rectangles enclosing
       those contours that verify the specified conditions (Enlarge them with the specified percentage)."""
    
    # Get original height and width, and adjust the gamma coefficient
    height, width, c = original_image.shape
    gamma = 1.5
    img = adjust_gamma(image=original_image, gamma=gamma)

    # convert the image to grayscale format
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find Canny edges
    edges = cv2.Canny(image=img_gray, threshold1=200, threshold2=350) # Increase the thresholds to keep less edges
    
    # Find contours using the findContours() function
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Eventually, show the results of these steps
    # plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    # plt.show()
    # Draw in blue the contours that were found
    # cv2.drawContours(image=edges, contours=contours, contourIdx=-1, color=(255, 0, 0), thickness=1)
    # plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
    # plt.show()
    
    # To eventually plot further results :
    # img_contour = original_image.copy()
    
    # To store interesting rectangles and special rectangles
    rectangles = []
    special_rectangles = []
    
    for cnt in contours:
        
        # Compute the perimeter of the contour and find a bounding box for it
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        augmented = False
        
        #If the given rectangle verifies the specified conditions, resize it and add it to the list of interesting rectangles
        if (conditions(w,h,minsize,peri)):
            
            # Resize the rectangle (enlarge it) 
            percent_w = w*percentage
            percent_h = h*percentage

            if (x - percent_w/2 < 0): x = 0
            else: x = int(x - percent_w/2)

            if (y - percent_h/2 < 0): y = 0
            else: y = int(y - percent_h/2)

            if (x + w + percent_w > width): w = width - x
            else: w = int(w + percent_w)

            if (y + h + percent_h > height): h = height - y
            else: h = int(h + percent_h)
            
            #To avoid resizing it a second time in the "special conditions"
            augmented = True
            
            #Add rectangle to the list of interesting rectangles
            rectangles.append((x,y,w,h))
            
            # Eventually, draw in blue the contours that are kept, and the bounding rectangle
            # cv2.drawContours(image=img_contour, contours=cnt, contourIdx=-1, color=(255, 0, 0), thickness=2)
            # cv2.rectangle(img_contour, (x,y), (x+w, y+h), (0, 0, 255), 2)
        
        
        #If the rectangle verifies the special conditions, resize it and add it to the list of special rectangles
        if(special_conditions[0](w,h,minsize,peri)):
            
            #If the rectangle has not yet been resized
            if not augmented :
                
                #Resize the rectangle
                percent_w = w*percentage
                percent_h = h*percentage
    
                if (x - percent_w/2 < 0): x = 0
                else: x = int(x - percent_w/2)
    
                if (y - percent_h/2 < 0): y = 0
                else: y = int(y - percent_h/2)
    
                if (x + w + percent_w > width): w = width - x
                else: w = int(w + percent_w)
    
                if (y + h + percent_h > height): h = height - y
                else: h = int(h + percent_h)
                    
            #Add rectangle to the list of special rectangles
            special_rectangles.append((x,y,w,h))
            
            # Eventually, draw in blue the contours that are kept, and the bounding rectangle
            # cv2.drawContours(image=img_contour, contours=cnt, contourIdx=-1, color=(255, 0, 0), thickness=2)
            # cv2.rectangle(img_contour, (x,y), (x+w, y+h), (0, 0, 255), 2)
    
    #Go through all special rectangles and see if two of them are interesting to merge
    for i in range(len(special_rectangles)):
        for j in range(i+1,len(special_rectangles)):
            #If they verify the conditions for merging, do so and add the result to the interesting rectangles
            if(special_conditions[1](special_rectangles[i],special_rectangles[j])):
                merged_rec = merge_rectangles(special_rectangles[i],special_rectangles[j])
                rectangles.append(merged_rec)
                
    # Eventually, visualise the contours that are kept, and their bounding rectangles
    # cv2.imshow("Window", img_contour)
    # cv2.waitKey(0)
    
    #Return the interesting rectangles
    return rectangles




#dictionary of labels: {TS_ID: TS_name, ...}
dico = {}
df = pd.read_excel("LEPL1507_TS.xlsx")
for index, row in df.iterrows():
    dico[row["c1"]] = row["c2"]
dico[62] = "Background"


#Load both models
pannelModel = models.load_model("my_background&panel_model")
cropped_model = models.load_model('my_balanced_model_merged_RMS_32_15_Augmented.h5')


#Load all images
X = []
image_nb = []
final_y_pred = []
folder_path = "data"
image_paths = os.listdir(folder_path)
for image_path in image_paths :
    if(image_path.endswith(".ppm") or image_path.endswith(".jpg")) :
        image_nb.append(image_path.split('.')[0])
        image = cv2.imread(folder_path + "/" + image_path)
        X.append(image)


# Predict class for each image
for image in X :
    
    # To store images classified as non-background
    kept_images = []
    probas_predictions = [] # To store probabilities of kept images
    class_predictions = []  # To store class predictions of kept images
    
    height, width, c = image.shape
    
    # Apply our three color filters
    red_image, red_conditions, red_spconds, red_percentage, red_minsize = extract_color(image, "red")
    blue_image, blue_conditions, blue_spconds, blue_percentage, blue_minsize = extract_color(image, "blue")
    orange_image, orange_conditions, orange_spconds, orange_percentage, orange_minsize = extract_color(image, "orange")
    
    
    # Find the contours associated with each filter
    red_rectangles = contours(red_image,red_conditions,red_spconds,red_percentage, red_minsize)
    blue_rectangles = contours(blue_image,blue_conditions, blue_spconds, blue_percentage, blue_minsize)
    orange_rectangles = contours(orange_image,orange_conditions, orange_spconds,orange_percentage, orange_minsize)
                
        
    # Merge all obtained rectangles and add the original image
    rectangles = red_rectangles + blue_rectangles + orange_rectangles
    rectangles.append((0,0,width,height))
    
    # Go through all obtained rectangles
    for rectangle in rectangles :
        
        x,y,w,h = rectangle
        
        # Get the cropped image corresponding to the rectangle
        # We copy it to avoid modifying original image
        cropped_image = image[y:y+h, x:x+w]
        clone = cropped_image.copy()
        
        # Resize image to pass it to our models
        img = cv2.resize(clone, (32, 32))
        img = img.astype("float32") / 255
        
        
        # Get predictions of TS + background model (background class == 62)
        pannel_predictions = pannelModel.predict(np.expand_dims(img, axis=0), verbose=0)
        idx2 = np.argmax(pannel_predictions.flatten())
        
        
        #If image is not classified as background, save the results for this image and the image itself
        if idx2 != 62 :
            probas_predictions.append(pannel_predictions[0][idx2])
            class_predictions.append(idx2)
            kept_images.append(img)
            
    # end rectangle for loop


    # probas_predictions now contains the probability for each kept sub_image
    # If empty, no sub_image was kept, so we return the cropped model prediction
    if probas_predictions == [] :
        img = cv2.resize(image, (32, 32))
        img = img.astype("float32") / 255
        cropped_predictions = cropped_model.predict(np.expand_dims(img, axis=0), verbose=0)
        best_prediction = np.argmax(cropped_predictions.flatten())
    
    # Otherwise, verify whether our two CNN obtain the same prediction for some of the kept sub_images or not
    else :
        found_same = False
        sames_proba = []
        sames_class = []
        cropped_probas = []
        cropped_classes = []
        
        # Used cropped model to predict the class of the kept sub_images
        kept_images = np.array(kept_images)
        cropped_predictions = cropped_model.predict(kept_images, verbose = 0)
        
        # Store the results
        for i in range(len(kept_images)) :
            idx = np.argmax(cropped_predictions[i].flatten())
            cropped_probas.append(cropped_predictions[i][idx])
            cropped_classes.append(idx)

            
        #Compare the result of TS + background model to that of the cropped model
        for i in range(len(kept_images)):
            
            # If both models have predicted the same output, keep this in mind and compute the sum of probabilities as final weight for this sub_image
            if(cropped_classes[i] == class_predictions[i]) :
                found_same = True
        
                # Ponderate the probas by multiplying results of both models
                sames_proba.append(cropped_probas[i] + probas_predictions[i])
                sames_class.append(cropped_classes[i])
        
        # If both models return same result for at least one sub_image, keep the image for wich the weight is highest
        if found_same :
            index = np.argmax(sames_proba)
            best_prediction = sames_class[index]
        
        # If no sub_image has same results on both models, keep the image with highest probability (no matter if it comes from the TS + backgound model or the cropped model)
        else :
            index = np.argmax(probas_predictions)
            proba = probas_predictions[index]
            best_prediction = class_predictions[index]
            index = np.argmax(cropped_probas)
            if cropped_probas[index] >= proba : best_prediction = cropped_classes[index]
    
    # Store the final prediction
    final_y_pred.append(best_prediction)


# If need be, destroy all windows
# cv2.destroyAllWindows()


# Put the results in the .csv file
df = pd.DataFrame()
df["Id"] = image_nb
df["Category"] = final_y_pred
df.to_csv("student.csv", index=False)
