from lib.utils import threshold_image, get_connected_components, plt_rectangles, divide_image
import cv2 as cv
import sys, os, shutil
import numpy as np
from matplotlib import pyplot as plt


''' THESE ARE PARAMETERS YOU CAN CHANGE  '''

thresholding_level = 25
min_side = 15
max_side = 200
border = 5
img_name = 'Original_Halved.tif'
batch_name = 'test'

'''             DOWN TO HERE            '''


''' This will ask you if you already have data in the output folder
given by batch_name'''

if not os.path.isdir(f'lib/imgs_np/{batch_name}'):
    os.makedirs(f'lib/imgs_np/{batch_name}')
    os.makedirs(f'lib/imgs_png/{batch_name}')
else:
    inputt = str(input(f'The folder {batch_name} already exists are you sure you want to continue? (y/n): ') or "y")
    if inputt=='n':
        sys.exit('User terminated as folder already exists')
    else:
        if not os.path.isdir(f'lib/imgs_png/{batch_name}'):
            os.makedirs(f'lib/imgs_png/{batch_name}')
        shutil.rmtree(f'lib/imgs_np/{batch_name}/')
        os.makedirs(f'lib/imgs_np/{batch_name}/')
        try:
            shutil.rmtree(f'lib/imgs_png/{batch_name}/')
            os.makedirs(f'lib/imgs_png/{batch_name}/')
        except:
            pass

'''
The main loop.
The if __name__ == '__main__' clause just checks that this is the script that is running (and nothing else is calling
this script by accident).
'''

if __name__=='__main__':

    '''This is where the real stuff happens'''

    # This reads the file img_name and outputs a numpy array for manipulation
    img_original = cv.imread(cv.samples.findFile(f"imgs/{img_name}",0))

    # This thresholds the image at the amount thresholding_level, with mode = 'mg' or 'bi' (mean gaussian or binary)
    a, img_thresholded = threshold_image(img_original, thresholding_level, mode = 'mg')

    # Takes the thresholded image and returns the rectangles around the particles
    stats = get_connected_components(img_thresholded, min_side=min_side, max_side=max_side, border = border)

    # Divides the ORIGINAL IMAGE according to stats and saves the fossils as numpy arrays in imgs_np.
    print('Saving individual fossils')
    divide_image(img_original, stats, batch_name, mode = 'np')
    divide_image(img_original, stats, batch_name, mode='png')

    # Plots rectangles on the thresholded images and saves the rectangles in imgs_rectangles
    print('Saving retangled image')
    img_rectangled = plt_rectangles(img_thresholded, stats)
    cv.imwrite(f'imgs_rectangled/{img_name.split(".")[0]}.png', img_rectangled)
    img_rectangled = plt_rectangles(img_original, stats)
    cv.imwrite(f'imgs_rectangled/{img_name.split(".")[0]}_original.png', img_rectangled)

    # This prints the number of fossils that were found
    print(stats.shape[0],'fossils were found')


