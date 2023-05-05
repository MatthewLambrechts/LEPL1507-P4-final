from skimage.feature import hog
import os
import cv2
from skimage import feature
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

     

images_train = []
labels_train = []

# get all the image folder paths
path_train = os.listdir(f"Training")
for label in path_train:
    if (label.split('.')[-1] == 'csv'):
            continue
    if (label.split('.')[-1] == 'txt'):
            continue
    if (label.split('.')[-1] == 'DS_Store'):
            continue
	# get all the image names
    all_images = os.listdir(f"Training/{label}")

	# iterate over the image names, get the label
    for image in all_images:
        if (image.split('.')[-1] != 'ppm'):
            continue

        image_path = f"Training/{label}/{image}"
        im = cv2.imread(image_path)
        image_resized = cv2.resize(im, (32, 32))

        # get the HOG descriptor for the image
        hog_desc = feature.hog(image_resized, orientations=30, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys', channel_axis=-1)
        
        # update the data and labels
        images_train.append(hog_desc)
        labels_train.append(label)
        

print('Training the classifier')
svm_model = LinearSVC(random_state=42, tol=1e-5)
svm_model.fit(images_train, labels_train)

#cv_scores = cross_val_score(svm_model, images_train, labels_train, cv=10)
#print("CV average score: %.2f" % cv_scores.mean())

print('Evaluating on test images')

# loop over the test dataset
path_test = os.listdir(f"eval_kaggle1_sorted")
correct_image = 0
number_image = 0

for label in path_test:

    if (label.split('.')[-1] == 'DS_Store'):
            continue
    
    lab = int(label)

	# get all the image names
    all_images = os.listdir(f"eval_kaggle1_sorted/{label}")

	# iterate over the image names, get the label
    for image in all_images:

        number_image += 1
        image_path = f"eval_kaggle1_sorted/{label}/{image}"
        #print(image_path)
        im = cv2.imread(image_path)
        image_resized = cv2.resize(im, (32, 32))

        # get the HOG descriptor for the image
        (hog_desc, hog_image) = feature.hog(image_resized, orientations=30, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys',visualize=True, channel_axis=-1)
        
    
        pred = svm_model.predict(hog_desc.reshape(1, -1))[0]
        
        if (int(pred) == lab):
             correct_image +=1
        #cv2.imshow('Test Image', im)
        #cv2.imshow('HOG Image', hog_image)
        #cv2.waitKey(0)

print("Accuracy : ", (correct_image/number_image)*100 )
