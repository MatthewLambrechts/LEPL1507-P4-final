""" Stratégie:

- sliding window pour diviser l'image en sous-images
- shape detection appliqué sur toute sous-image: uniquement les sous-images contenant une shape sont gardées
- pour toute sous-image contenant une shape: application du modèle simple pour images croppées (pour l'instant sans bruit)
- on prend la prédiction de plus grande proba sur toutes les sous-images
- (les sous-images dans lesquelles une shape est détectée différente d'un panneau auront des prédictions de proba faible une fois le modèle appliqué)

Dernière contrainte à prendre en compte: le bruit
Même stratégie sauf qu'on appliquera sur chaque sous-image contenant une shape le modèle simple d'images croppées avec bruit

Modèle simple d'images croppées avec bruit = modèle simple d'images croppées sans bruit + data augmentation pour chaque classe avec images bruitées

Ci-dessous:
Code source de shape detection à appliquer sur chaque sous-image obtenu avec sliding window

Shapes possibles des panneaux: triangle, cercle, octogone, rectangle
"""


import cv2
import numpy as np
from matplotlib import pyplot as plt

# Fonction pour la visualisation de la détection d'un des contours intéressants
def visualize_detection(img, contour, name_of_shape):
	cv2.drawContours(img, [contour], 0, (255, 255, 0), 5) #cyan
	cv2.putText(img, name_of_shape, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2) #jaune
	print(name_of_shape)

img = cv2.imread("Meta Belgium/40.png") #exemple simple de triangle

# Convertit l'image en Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Effectue un threshold sur l'image
# Les contours sont dans la direction du gradient de couleurs (noir&blanc)
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Liste des noms des formes
contours, _ = cv2.findContours(
	threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

i = 0

# Une image contient différents contours (courbes) fermés, ici on parcourt tous les contours que l'algo a trouvés dans l'image jusqu'à ce qu'on détecte un contour qui nous intéresse
#les contours qui nous intéressent: triangle, cercle, octogone, rectangle
#même si l'image contient différents contours intéressant, on arête l'algo dès qu'un contour intéressant a été trouvé  
#et on place l'image dans le numpy array des images contenant un contour intéressant: array à créer dans SlidingWindow
print("The important shape that has been detected:")
for contour in contours:

	# Ici on ignore le premier contour parce que findContour détecte toute l'image comme contour
	# Image non-intéressante car non-croppée
	if i == 0:
		i = 1
		continue

	# Fonction cv2.approxPolyDP() pour approximer la forme
	approx = cv2.approxPolyDP(
		contour, 0.01 * cv2.arcLength(contour, True), True)

	# Trouve le centre de la forme
	M = cv2.moments(contour)
	if M['m00'] != 0.0:
		x = int(M['m10']/M['m00'])
		y = int(M['m01']/M['m00'])

    # if len(approx) == 3, 4, 8 ou else: on garde l'image 
	# On met le nom de la forme au centre de la forme
	# Argument BGR de cv2.putText est un tuple de taille 3 (.,.,.) contenant une des combinaisons possibles
	# R,G et B: [0,255]
	# Voir le tableau RGB (RVB en français) du système de couleurs RGB
	# Attention, ici: l'ordre est BGR et non RGB
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

	# Dangereux: détecte plus que de simples cercles 
	# Quel est la taille de "approx" si on détecte un cercle ?
	# -> Faiblesse de l'algo
	else:
		visualize_detection(img, contour, "Circle")

		#TODO: ajouter img dans array

		break


# On montre la visualisation de la détection: contour + nom de la forme géométrique
cv2.imshow('shapes', img)

# On attend jusqu'à ce qu'on arrête le programme
cv2.waitKey(0)

# Une fois qu'on arrête le programme, toutes les fenêtres contenant les images avec contours dessinés sont fermées
cv2.destroyAllWindows()



