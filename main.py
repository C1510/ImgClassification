import copy

from lib.utils import threshold_image, get_connected_components, plt_rectangles, divide_image, cut_out_of_image
import cv2 as cv
import sys, os, shutil
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


''' THESE ARE PARAMETERS YOU CAN CHANGE  '''

thresholding_level = 25
min_side = 15
max_side = 200
border = 5
img_name = 'Original_Halved.tif'
batch_name = 'test'
threshold_mode = 'mg' #'mg' or 'bi'
save_noclass = True

'''             DOWN TO HERE            '''


''' This will ask you if you already have data in the output folder
given by batch_name'''

img_name_no_ext = img_name.split('.')[0]
if not os.path.isdir(f'imgs_rectangled/{batch_name}_{img_name_no_ext}'):
    os.makedirs(f'imgs_rectangled/{batch_name}_{img_name_no_ext}')
else:
    inputt = str(input(f'The folder {batch_name}_{img_name_no_ext} already exists are you sure you want to continue? (y/n): ') or "y")
    if inputt=='n':
        sys.exit('User terminated as folder already exists')
    else:
        shutil.rmtree(f'imgs_rectangled/{batch_name}_{img_name_no_ext}')
        os.makedirs(f'imgs_rectangled/{batch_name}_{img_name_no_ext}')

def save_classified_images_no_classification(stats, img):
    # This function takes the stats file and an image, and saves your classifications to the
    # imgs_classified and imgs_classified_png folders.
    for c, col in stats.iterrows():
        # Saves file according to classification
        if not os.path.isdir(f'imgs_classified_png/{batch_name}_{img_name_no_ext}_noclass/-1/'):
            os.makedirs(f'imgs_classified_png/{batch_name}_{img_name_no_ext}_noclass/-1/')
        cv.imwrite(f'imgs_classified_png/{batch_name}_{img_name_no_ext}_noclass/-1/{c}.png', cut_out_of_image(img, col))
    return

'''
The main loop.
The if __name__ == '__main__' clause just checks that this is the script that is running (and nothing else is calling
this script by accident).
'''

if __name__=='__main__':

    '''This is where the real stuff happens'''

    # This reads the file img_name and outputs a numpy array for manipulation
    img_original = cv.imread(cv.samples.findFile(f"imgs/{img_name}",0))
    img_original_copy = copy.deepcopy(img_original)

    # This thresholds the image at the amount thresholding_level, with mode = 'mg' or 'bi' (mean gaussian or binary)
    a, img_thresholded = threshold_image(img_original, thresholding_level, mode = threshold_mode)

    # Takes the thresholded image and returns the rectangles around the particles
    stats = get_connected_components(img_thresholded, min_side=min_side, max_side=max_side, border = border)

    # Divides the ORIGINAL IMAGE according to stats and saves the fossils as numpy arrays in imgs_np.
    # CURRENTLY UNUSED IN THIS ITERATION
    # print('Saving individual fossils')
    # divide_image(img_original, stats, batch_name, mode = 'np')
    # divide_image(img_original, stats, batch_name, mode = 'png')

    # Plots rectangles on the thresholded images and saves the rectangles in imgs_rectangles/{batch_name}_{image_name_no_ext}
    print('Saving retangled image')
    img_rectangled = plt_rectangles(img_thresholded, stats)
    cv.imwrite(f'imgs_rectangled/{batch_name}_{img_name_no_ext}/{img_name.split(".")[0]}.png', img_rectangled)
    img_rectangled = plt_rectangles(img_original, stats)
    cv.imwrite(f'imgs_rectangled/{batch_name}_{img_name_no_ext}/{img_name.split(".")[0]}_original.png', img_rectangled)

    # This prints the number of fossils that were found
    print(stats.shape[0],'fossils were found')

    #Adds an extra column to stats so we can put classifications in later. -1 represents unclassified values
    stats = np.c_[stats, (-1+np.zeros(stats.shape[0]).reshape(-1,1))]

    # Saves stats in human-readable form to imgs_rectangles/{batch_name}_{image_name_no_ext}
    np.savetxt(f'imgs_rectangled/{batch_name}_{img_name_no_ext}/{img_name.split(".")[0]}.txt', stats, fmt='%.0f', delimiter=' ', header='left_top_x left_top_y x_length y_length vol class',comments='')

    if save_noclass:
        stats2 = pd.read_csv(f'imgs_rectangled/{batch_name}_{img_name_no_ext}/{img_name.split(".")[0]}.txt', delimiter=' ')
        save_classified_images_no_classification(stats2, img_original_copy)

