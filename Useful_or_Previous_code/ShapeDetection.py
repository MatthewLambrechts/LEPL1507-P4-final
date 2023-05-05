""" Stratégie:

- sliding window pour diviser l'image en sous-images
- shape detection appliqué sur toute sous-image: uniquement les sous-images contenant une shape sont gardées
- pour toute sous-image contenant une shape: application du modèle simple pour images croppées (pour l'instant sans bruit)
- on prend la prédiction de plus grande proba sur toutes les sous-images
- (les sous-images dans lesquelles une shape est détectée différente d'un panneau auront des prédictions de proba faible une fois le modèle appliqué)

dernière contrainte à prendre en compte: le bruit
même stratégie sauf qu'on appliquera sur chaque sous-image contenant une shape le modèle simple d'images croppées avec bruit

modèle simple d'images croppées avec bruit = modèle simple d'images croppées sans bruit + data augmentation pour chaque classe avec images bruitées

Ci-dessous:
code source de shape detection à appliquer sur chaque sous-image obtenu avec sliding window

shapes possibles des panneaux: triangle, cercle, octogone, rectangle
"""


import cv2
import numpy as np
from matplotlib import pyplot as plt

#fonction pour la visualisation de la détection d'un des contours intéressants
def visualize_detection(img, contour, name_of_shape):
	# using drawContours() function
	cv2.drawContours(img, [contour], 0, (255, 255, 0), 5) #cyan
	cv2.putText(img, name_of_shape, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2) #jaune
	print(name_of_shape)


#code à introduire dans la boucle du window dans SlidingWindow.py: @Roy, à toi l'honneur

# reading image
# pour chaque image entourée par le window
img = cv2.imread("Meta Belgium/40.png") #exemple simple de triangle

# converting image into grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# setting threshold of gray image
# les contours son dans la direction du gradient de couleurs (noir&blanc)
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

#using a findContours() function
#list for storing names of shapes
contours, _ = cv2.findContours(
	threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

i = 0

#une image contient différents contours (courbes) fermés, ici on parcourt tous les contours que l'algo a trouvés dans l'image jusqu'à ce qu'on détecte un contour qui nous intéresse
#les contours qui nous intéressent: triangle, cercle, octogone, rectangle
#même si l'image contient différents contours intéressant, on arête l'algo dès qu'un contour intéressant a été trouvé  
#et on place l'image dans le numpy array des images contenant un contour intéressant: array à créer dans SlidingWindow
print("The important shape that has been detected:")
for contour in contours:

	# here we are ignoring first contour because
	# findcontour function detects whole image as shape
	# image non-intéressante car non-croppée
	if i == 0:
		i = 1
		continue

	# cv2.approxPolyDP() function to approximate the shape
	approx = cv2.approxPolyDP(
		contour, 0.01 * cv2.arcLength(contour, True), True)

	# finding center point of shape
	M = cv2.moments(contour)
	if M['m00'] != 0.0:
		x = int(M['m10']/M['m00'])
		y = int(M['m01']/M['m00'])

    #if len(approx) == 3, 4, 8 ou else: on garde l'image 
	# putting shape name at center of each shape
	#argument BGR de cv2.putText est un tuple de taille 3 (.,.,.) contenant une des combinaisons possibles
	#R,G et B: [0,255]
	#voir le tableau RGB (RVB en français) du système de couleurs RGB
	#attention, ici: l'ordre est BGR et non RGB
	if len(approx) == 3:
		visualize_detection(img, contour, "Triangle")

		#TODO: ajouter img dans array

		break

	elif len(approx) == 4:
		visualize_detection(img, contour, "Quadrilateral")

		#TODO: ajouter img dans array

		break
		
	elif len(approx) == 8:
		visualize_detection(img, contour, "Octogon")

		#TODO: ajouter img dans array

		break

	#dangereux: détecte plus que de simples cercles 
	#quel est la taille de "approx" si on détecte un cercle ?
	#faiblesse de l'algo
	else:
		visualize_detection(img, contour, "Circle")

		#TODO: ajouter img dans array

		break


#On montre la visualisation de la détection: contour + nom de la forme géométrique

# displaying the image after drawing contours
cv2.imshow('shapes', img)

#on attend jusqu'à ce qu'on arrête le programme
cv2.waitKey(0)

#une fois qu'on arrête le programme, toutes les fenêtres contenant les images avec contours dessinés sont fermées
cv2.destroyAllWindows()



