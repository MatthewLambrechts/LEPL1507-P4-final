import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


#on ne déplace pas les images mais on les copie (ainsi, on garde une trace des anciens dossiers)

""" 
- créer les 63 classes dans Background&PanelTrainingSet et Background&PanelTestingSet [-1, 61]: done
- copier 70% de BackgroundImages dans classe -1 de Background&PanelTrainingSet: done 
- copier AugmentedBelgianTrainingSet dans les classes correspondantes de Background&PanelTrainingSet: done
- copier 30% de BackgroundImages dans classe -1 de Background&PanelTestingSet: done
- copier BelgianTestingSet dans les classes correspondantes de Background&PanelTestingSet: done

Ainsi, on a créé nos Training et Test sets

ordre des classes dans Background&panelTrainingSet et Background&panelTestingSet: 0, -1, 1, 2, 3, ... !!!
 """


#création des  63 dossiers (classes [-1, 61]) dans Background&PanelTrainingSet et Background&PanelTestingSet

for i in range(-1, 62):
    #pour être compatible avec les autres dossiers: AugmentedBelgianTrainingSet et BelgiantestingSet
    if(i >= -1 and i <= 9):
        os.makedirs('Background&PanelTrainingSet/' + "0000" + str(i))
        os.makedirs('Background&PanelTestingSet/' + "0000" + str(i))

    else:
        os.makedirs('Background&PanelTrainingSet/' + "000" + str(i))
        os.makedirs('Background&PanelTestingSet/' + "000" + str(i))


#remplir Background&PanelTestingSet et Background&PanelTrainingSet

#AugmentedBelgianTrainingSet -> Background&PanelTrainingSet classe i
#BelgianTestingSet -> Background&PanelTestingSet classe i
#dir_source, dir_destination = str
#split = float in [0,1]
#return: /
def transfer_panels_sep(dir_source, dir_train_destination, dir_test_destination, split):
    image_size=32
    folders = sorted(os.listdir(dir_source))

    #parcours des classes
    for folder_path in folders:
        if folder_path == "Readme.txt":
            continue #on nie

        X = []

        # parcours des images de la classe
        for file_path in os.listdir(dir_source+ "/" + folder_path):
            #on ne lit que les images, pas les .csv de chaque classe contenant les coordonnées des sommets du rectangle correcte de la détection et la classID correcte pour entraîner le modèle
            if os.path.splitext(file_path)[-1] == '.ppm':
                image = cv2.imread(dir_source + "/" + folder_path + "/" + file_path)

                # X contient les images resized à la taille 32
                X.append(cv2.resize(image, (image_size, image_size)))

        #images ds la classe courante
        nb_images = len(X)
        permut = np.random.permutation(nb_images)
        
        X = np.array(X)
        X = X[permut]

        # save toutes les images de la classe courante dans la classe correspondante de dir_destination
        for i in range(int(nb_images*split)):
            plt.imsave(dir_train_destination + "/" + folder_path + "/" + str(i) + ".ppm", cv2.cvtColor(X[i], cv2.COLOR_BGR2RGB))
        for i in range(int(nb_images*split),nb_images):
            plt.imsave(dir_test_destination + "/" + folder_path + "/" + str(i) + ".ppm", cv2.cvtColor(X[i], cv2.COLOR_BGR2RGB))

    return


# 70% BackgroundImages -> Background&PanelTrainingSet classe -1
# 30% BackgroundImages -> Background&PanelTestingSet classe -1
#dir_source, dir_destination = str ; begin_idx_file, end_idx_file = respectivement index du premier et dernier élément (non-compris) dans la liste "files" des fichiers du répertoire dir_source (ici:"BackgroundImages")
#on choisit (~30%) pour le testing set et (~70%) pour le training set
def transfer_backgrounds(dir_source, dir_destination, nbr_files, begin_idx_file, end_idx_file, seed):

    image_size=32

    #os.listdir() renvoit une liste contenant tous les fichiers et répertoires dans dir_source (ici: que des fichiers) dans un ordre arbitraire
    #on utilise sorted() pour avoir les fichiers dans l'ordre d'apparition dans dir_source parce qu'on veut de l'ordre car on doit appeler deux fois la fonction pour des ratios différents (30%, 70%)
    files = sorted(os.listdir(dir_source))
    
    idx = np.random.RandomState(seed=seed).permutation(nbr_files)
    
    X = []
    #Y = []
    # parcours des index de files
    for idx_file_path in idx[begin_idx_file : end_idx_file]:
        #on ne lit que les .ppm
        if os.path.splitext(files[idx_file_path])[-1] == '.ppm':
            image = cv2.imread(dir_source + "/" + files[idx_file_path])

            # X contient les images resized à la taille 32
            X.append(cv2.resize(image, (image_size, image_size)))
            #vecteur constant, contient le label de la classe pour chaque image (-1,...,-1) puis (0,...,0) puis (1,...,1), ...
            #Y.append(label)

    #images ds la classe courante
    nb_images = len(X)
    
    X = np.array(X)
    #Y = np.array(Y)

    for i in range(nb_images):
        plt.imsave(dir_destination + "/" + "0000-1" + "/" + str(i) + ".ppm", cv2.cvtColor(X[i], cv2.COLOR_BGR2RGB))
    
    return

####################################################################################################################################
#remplissage de Background&PanelTrainingSet et Background&PanelTestingSet


# #AugmentedBelgianTrainingSet -> Background&PanelTrainingSet classes [0,61]
# transfer_panels_sep("AugmentedBelgianTrainingSet", "Background&PanelTrainingSet")

# #BelgianTestingSet -> Background&PanelTestingSet classes [0,61]
# transfer_panels("BelgianTestingSet", "Background&PanelTestingSet")

#Merged_and_Balanced_DataSet -> Background&PanelTrainingSet, Background&PanelTestingSet - classes [0,61]
transfer_panels_sep("Merged_and_Balanced_DataSet", "Background&PanelTrainingSet", "Background&PanelTestingSet", 0.7)

files = sorted(os.listdir("Background")) #return list
nb_files = len(files)
seed = 42

# 70% BackgroundImages -> Background&PanelTrainingSet classe -1
transfer_backgrounds("Background", "Background&PanelTrainingSet", nb_files, 0, int(nb_files*0.7), seed) #int() renvoit partie entière!

# 30% BackgroundImages -> Background&PanelTestingSet classe -1
transfer_backgrounds("Background", "Background&PanelTestingSet", nb_files, int(nb_files*0.7), nb_files, seed)










