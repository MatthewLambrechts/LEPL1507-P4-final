import cv2
import matplotlib.pyplot as plt
import os
import numpy as np



def pyramid(image, scale=1.5, minSize=(5, 5)):
    it = 0
    m = min(image.shape[0]*scale, image.shape[1]*scale)
    while it < 8:
  		  # Calcule les nouvelles dimensions de l'image et on la redimensionne
          m = int(m/scale)
          it += 1
          
  		  # Si elle est plus petite que le minimum requis on arrête
          if m < minSize[0] or m < minSize[1]:
              break
          yield (m,m)
          

# Génère une sous image random d'une certaine taille
def sliding_window_2(image, windowSize):
    x = np.random.randint(0,image.shape[1]-windowSize[1]+1)
    y = np.random.randint(0,image.shape[0]-windowSize[0]+1)
    yield (x, y, image[y:y + windowSize[0], x:x + windowSize[1]])



# On sauvegarde le nouveau dataset d'images de background
path = "NonTSImages/TrainingBG"
save_path = "Backround"
folder = os.listdir(path)
i = 0
for file in folder :
    j = 0
    if i%1 == 0 :
        image = cv2.imread(os.path.join(path,file))
        for windowSize in pyramid(image, scale=1.5):
            # Boucle sur le sliding window sur chaque étage de la pyramide
            for (x, y, window) in sliding_window_2(image, windowSize):
                im = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
                im = cv2.resize(im,(32,32))
                plt.imsave(f"Backround/{i}{j}.ppm",im)
                j+=1    
    i += 1