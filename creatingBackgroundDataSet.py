# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 09:12:21 2023

@author: Valentin
"""

import cv2
import matplotlib.pyplot as plt
import os
import numpy as np



def pyramid(image, scale=1.5, minSize=(5, 5)):
	# yield the original image
    it = 0
    m = min(image.shape[0]*scale, image.shape[1]*scale)
    while it < 8:
  		  # compute the new dimensions of the image and resize it
          m = int(m/scale)
          it += 1
  		  # if the resized image does not meet the supplied minimum
  		  # size, then stop constructing the pyramid
          if m < minSize[0] or m < minSize[1]:
              break
          yield (m,m)
          

#Yield a random sub_image of this size
def sliding_window_2(image, windowSize):
    x = np.random.randint(0,image.shape[1]-windowSize[1]+1)
    y = np.random.randint(0,image.shape[0]-windowSize[0]+1)
    yield (x, y, image[y:y + windowSize[0], x:x + windowSize[1]])




path = "NonTSImages/TrainingBG"
save_path = "Backround"
folder = os.listdir(path)
i = 0
for file in folder :
    j = 0
    if i%1 == 0 :
        image = cv2.imread(os.path.join(path,file))
        for windowSize in pyramid(image, scale=1.5):
            # loop over the sliding window for each layer of the pyramid
            for (x, y, window) in sliding_window_2(image, windowSize):
                im = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
                im = cv2.resize(im,(32,32))
                plt.imsave(f"Backround/{i}{j}.ppm",im)
                j+=1
                # plt.imshow(cv2.cvtColor(window, cv2.COLOR_BGR2RGB))
                # plt.show()
            
    i += 1


#"NonTSImages/TrainingBG/image.000010.c00.jp2"