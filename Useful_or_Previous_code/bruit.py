"""
LEPL1507 - Projet 4
Auteurs : Groupe 3

Code permettant de simuler des conditions météorologiques en appliquant des filtres sur les panneaux de circulation

Ce code est fortement inspiré des deux sources suivantes :
- https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
- https://github.com/AmardeepSarang/AutoSnow-A-synthetic-winter-image-generator-framework
"""

import os
import cv2
import math
import random
import numpy as np
import matplotlib.pyplot as plt



def is_list(x):
    return type(x) is list


###################### HLS ######################

def hls(image,src='RGB'):
    if(is_list(image)):
        image_HLS=[]
        image_list=image
        for img in image_list:
            eval('image_HLS.append(cv2.cvtColor(img,cv2.COLOR_'+src.upper()+'2HLS))')
    else:
        image_HLS = eval('cv2.cvtColor(image,cv2.COLOR_'+src.upper()+'2HLS)')
    return image_HLS





###################### BGR ######################

def bgr(image, src='RGB'):
    if(is_list(image)):
        image_BGR=[]
        image_list=image
        for img in image_list:
            eval('image_BGR.append(cv2.cvtColor(img,cv2.COLOR_'+src.upper()+'2BGR))')
    else:
        image_BGR= eval('cv2.cvtColor(image,cv2.COLOR_'+src.upper()+'2BGR)')
    return image_BGR





###################### RGB ######################

def rgb(image, src='BGR'):
    if(is_list(image)):
        image_RGB=[]
        image_list=image
        for img in image_list:
            eval('image_RGB.append(cv2.cvtColor(img,cv2.COLOR_'+src.upper()+'2RGB))')
    else:
        image_RGB= eval('cv2.cvtColor(image,cv2.COLOR_'+src.upper()+'2RGB)')
    return image_RGB





###################### Change the brightness ######################

def change_light(image, coeff):
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS
    image_HLS = np.array(image_HLS, dtype = np.float64) 
    image_HLS[:,:,1] = image_HLS[:,:,1]*coeff ## scale pixel values up or down for channel 1(Lightness)
    if(coeff>1):
        image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255 ##Sets all values above 255 to 255
    else:
        image_HLS[:,:,1][image_HLS[:,:,1]<0]=0
    image_HLS = np.array(image_HLS, dtype = np.uint8)
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
    return image_RGB 


def brighten(image, brightness_coeff=-1): ##function to brighten the image
    if(brightness_coeff == -1):
        brightness_coeff_t = 1 + random.uniform(0,1) ## coeff between 1.0 and 1.5
    else:
        brightness_coeff_t = 1+ brightness_coeff ## coeff between 1.0 and 2.0
    image_RGB= change_light(image, brightness_coeff_t)
    return image_RGB


def darken(image, darkness_coeff=-1): ##function to darken the image
    if(darkness_coeff == -1):
            darkness_coeff_t = 1- random.uniform(0,1)
    else:
        darkness_coeff_t = 1 - darkness_coeff  
    image_RGB= change_light(image,darkness_coeff_t)
    return image_RGB





###################### SNOW ######################

def adjust_brightness(image,brightness_coefficient = 0.7):
    image_HLS = am.hls(image) ## Conversion to HLS
    image_HLS[:,:,1] = image_HLS[:,:,1]*brightness_coefficient ## scale pixel values down for channel 1(Lightness)
    image_RGB= am.rGb(image_HLS,'hls')
    return image_RGB


def vertical_motion_blur(img, kernel_size=7):
    # Create the vertical kernel.
    kernel_v = np.zeros((kernel_size, kernel_size))

    # Fill the middle row with ones.
    kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
    # Normalize.
    kernel_v /= kernel_size
    
    # Apply the vertical kernel.
    vertical_mb = cv2.filter2D(img, -1, kernel_v)
    return vertical_mb


def horizontal_motion_blur(img, kernel_size=7):
    # Create the kernel.
    kernel_h = np.zeros((kernel_size, kernel_size))

    # Fill the middle row with ones.
    kernel_h[int((kernel_size - 1)/2),:] = np.ones(kernel_size)
    # Normalize.
    kernel_h /= kernel_size
    
    # Apply the vertical kernel.
    horizontal_mb = cv2.filter2D(img, -1, kernel_h)
    return horizontal_mb


def add_snow_noise(img, scale_percent=300, flakes_amount_threshold=0.4, motion_blur_amount=7, ground_snow=False, blur_type='vb'):
    scale_percent = np.random.randint(500, 1000)
    flakes_amount_threshold = np.random.uniform(0.2, 0.3)
    
    # Create noise the imitate fine snow
    area=img.shape[0]*img.shape[1]
    noise = np.random.choice([0, 255], size=area, p=[1-flakes_amount_threshold, flakes_amount_threshold])
    noise=np.reshape(noise,(img.shape[0],img.shape[1])).astype('float32')

    snow_layer = np.dstack((noise,noise,noise))
    # Percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Resize image to change size of snow flakes
    snow_layer = cv2.resize(snow_layer, dim, interpolation=cv2.INTER_AREA)
    snow_layer=snow_layer[0:img.shape[0],0:img.shape[1]]
    snow_layer=snow_layer.astype(np.uint8)
    
    #add motion blur
    if blur_type=='v':
        snow_layer=vertical_motion_blur(snow_layer,motion_blur_amount)
    elif blur_type=='h':
        snow_layer=horizontal_motion_blur(snow_layer,motion_blur_amount)
    elif blur_type=="vb":
        snow_layer=vertical_motion_blur(snow_layer,motion_blur_amount)
        snow_layer=horizontal_motion_blur(snow_layer,motion_blur_amount)
    
    # Blend snow with images
    snow_image = cv2.addWeighted(snow_layer,1,img,1,0)
    return snow_image





###################### RAIN ######################

def generate_random_lines(imshape, slant, no_of_drops, drop_length):
    drops = []
    for i in range(no_of_drops):
        # Random x position (attention à l'inclinaison de la ligne)
        if slant < 0:
            x = np.random.randint(-slant, imshape[1])
        else:
            x = np.random.randint(0, imshape[1] - slant)
        y = np.random.randint(0, imshape[0] - drop_length) # Random y position start of rain drop minus its length
        drops.append((x,y))
    return drops, drop_length


def rain_process(image, slant, drop_length, drop_color, drop_width, rain_drops):
    imshape = image.shape
    image_t = image.copy()
    for rain_drop in rain_drops:
        cv2.line(image_t, (rain_drop[0], rain_drop[1]), (rain_drop[0]+slant, rain_drop[1]+drop_length), drop_color, drop_width)
    image = cv2.blur(image_t, (2,2)) # Rainy view are blurry
    brightness_coefficient = 1.0 # Rainy days are usually shady 
    image_HLS = hls(image) # Conversion to HLS
    image_HLS[:,:,1] = image_HLS[:,:,1]*brightness_coefficient # Scale pixel values down for channel 1 (Lightness)
    image_RGB = rgb(image_HLS, 'hls') # Conversion to RGB
    return image_RGB


def add_rain(image, drop_width=1, drop_color=(200,200,200)): # (200,200,200) a shade of gray
    slant = np.random.randint(-50,50) # Generate random slant (inclinaison des lignes)
    # print(slant)
    no_of_drops = np.random.randint(500, 1000) # If You want heavy rain, try increasing this
    drop_length = np.random.randint(50, 100)
    imshape = image.shape
    rain_drops, drop_length = generate_random_lines(imshape, slant, no_of_drops, drop_length)
    output = rain_process(image, slant, drop_length, drop_color, drop_width, rain_drops)
    image_RGB = output
    return image_RGB





########## Ajout d'images avec des filtres simulant des conditions météorologiques ##########

folder = "Datasets\\Belgium Dataset\\BelgiumTSC_Training\\Training\\"

amount = 15
final_size = (128,128)

for folder_path in os.listdir(folder):
    if folder_path.split(".")[-1] == "txt":
        continue

    array = []
    for image_nb in os.listdir(folder + folder_path):
        if image_nb.split(".")[-1] == "csv":
            continue
        array.append(image_nb)

    print(folder_path)
    label = int(folder_path)
    image = 375

    list = np.random.choice(range(0, len(array)), size=amount)
    for index in list:
        print(array[index])
        # Opening image
        img = cv2.imread(folder + folder_path + "\\" + str(array[index]))
        coeff = random.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        print(f"coeff {coeff}")
        img_brighten = brighten(img, brightness_coeff=coeff)
        img_resize = cv2.resize(img_brighten, final_size)
        cv2.imwrite(f'Datasets\\Bruit\\{label}\\{image}.ppm', img_resize)
        cv2.imwrite(f'Datasets\\BalancedDataSet\\{folder_path}\\{image}.ppm', img_resize)
        # cv2.imshow("img", cv2.resize(img_brighten, (500,500)))
        # cv2.waitKey(1000)
        image += 1


    list = np.random.choice(range(0, len(array)), size=amount)
    for index in list:
        print(array[index])
        # Opening image
        img = cv2.imread(folder + folder_path + "\\" + str(array[index]))
        coeff = random.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.75])
        print(f"coeff {coeff}")
        img_dark = darken(img, darkness_coeff=coeff)
        img_resize = cv2.resize(img_dark, final_size)
        cv2.imwrite(f'Datasets\\Bruit\\{label}\\{image}.ppm', img_resize)
        cv2.imwrite(f'Datasets\\BalancedDataSet\\{folder_path}\\{image}.ppm', img_resize)
        #cv2.imshow("img", cv2.resize(img_dark, (500,500)))
        #cv2.waitKey(1000)
        image += 1

    list = np.random.choice(range(0, len(array)), size=amount)
    for index in list:
        print(array[index])
        # Opening image
        img = cv2.imread(folder + folder_path + "\\" + str(array[index]))
        img_resize = cv2.resize(img, (1000,1000))
        img_snow = add_snow_noise(img_resize, scale_percent=200, flakes_amount_threshold=0.3, motion_blur_amount=7, ground_snow=False)
        img_resize = cv2.resize(img_snow, final_size)
        cv2.imwrite(f'Datasets\\Bruit\\{label}\\{image}.ppm', img_resize)
        cv2.imwrite(f'Datasets\\BalancedDataSet\\{folder_path}\\{image}.ppm', img_resize)
        # cv2.imshow("img", cv2.resize(img_snow, (500,500)))
        # cv2.waitKey(1000)
        image += 1

    list = np.random.choice(range(0, len(array)), size=amount)
    for index in list:
        print(array[index])
        # Opening image
        img = cv2.imread(folder + folder_path + "\\" + str(array[index]))
        img_resize = cv2.resize(img, (500,500))
        img_rain = add_rain(img_resize)
        img_resize = cv2.resize(img_rain, final_size)
        cv2.imwrite(f'Datasets\\Bruit\\{label}\\{image}.ppm', img_resize)
        cv2.imwrite(f'Datasets\\BalancedDataSet\\{folder_path}\\{image}.ppm', img_resize)
        # cv2.imshow("img", cv2.resize(img_rain, (500,500)))
        # cv2.waitKey(1000)
        image += 1





########## Example with one image ##########

# from keras import models
# model_name = 'Moi/my_balanced_model_merged_RMS_32_15_Augmented.h5'
# model = models.load_model(model_name)

folder_example = "Datasets\\Bruit exemple"

path = "Datasets\\Belgium Dataset\\BelgiumTSC_Training\\Training\\00000\\01153_00000.ppm"
img = cv2.imread(path)
plt.imshow(cv2.cvtColor(cv2.resize(img, (500,500)), cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.savefig(f"{folder_example}\\Original_image.pdf", format="pdf")
# cv2.imshow("img", cv2.resize(img, (500,500)))
# cv2.waitKey(0)


# Brighten
for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    print(f"coeff {i}")
    img_brighten = brighten(img, brightness_coeff=i)
    img_resize = cv2.resize(img_brighten, (32,32))
    # predictions = model.predict(np.expand_dims(img_resize.astype("float32") / 255, axis=0))
    # print(np.argmax(predictions))
    plt.imshow(cv2.cvtColor(cv2.resize(img_brighten, (500,500)), cv2.COLOR_BGR2RGB))
    plt.savefig(f"{folder_example}\\Brighten_{i}.pdf", format="pdf")
    # cv2.imshow("img", cv2.resize(img_brighten, (500,500)))
    # cv2.waitKey(0)

# Darkness
for i in [0.1, 0.2, 0.3, 0.4, 0.5]:
    print(f"coeff {i}")
    img_dark = darken(img, darkness_coeff=i)
    img_resize = cv2.resize(img_dark, (32,32))
    # predictions = model.predict(np.expand_dims(img_resize.astype("float32") / 255, axis=0))
    # print(np.argmax(predictions))
    plt.imshow(cv2.cvtColor(cv2.resize(img_dark, (500,500)), cv2.COLOR_BGR2RGB))
    plt.savefig(f"{folder_example}\\Dark_{i}.pdf", format="pdf")
    # cv2.imshow("img", cv2.resize(img_dark, (500,500)))
    # cv2.waitKey(0)

# Snow
for i in range(10):
    print(f"iter {i}")
    img = cv2.imread(path)
    img_resize = cv2.resize(img, (1000,1000))
    img_snow = add_snow_noise(img_resize, scale_percent=300, flakes_amount_threshold=0.4, motion_blur_amount=7, ground_snow=False, blur_type='vb') 
    img_resize = cv2.resize(img_snow, (32,32))
    # predictions = model.predict(np.expand_dims(img_resize.astype("float32") / 255, axis=0))
    # print(np.argmax(predictions))
    plt.imshow(cv2.cvtColor(cv2.resize(img_snow, (500,500)), cv2.COLOR_BGR2RGB))
    plt.savefig(f"{folder_example}\\Snow_{i}.pdf", format="pdf")
    # cv2.imshow("img", cv2.resize(img_snow, (500,500)))
    # cv2.waitKey(0)

# Rain
for i in range(10):
    print(f"iter {i}")
    img = cv2.imread(path)
    img_resize = cv2.resize(img, (500,500))
    img_rain = add_rain(img_resize)
    img_resize = cv2.resize(img_rain, (32,32))
    # predictions = model.predict(np.expand_dims(img_resize.astype("float32") / 255, axis=0))
    # print(np.argmax(predictions))
    plt.imshow(cv2.cvtColor(cv2.resize(img_rain, (500,500)), cv2.COLOR_BGR2RGB))
    plt.savefig(f"{folder_example}\\Rain_{i}.pdf", format="pdf")
    # cv2.imshow("img", cv2.resize(img_rain, (500,500)))
    # cv2.waitKey(0)