import pandas as pd

from lib.utils import threshold_image, get_connected_components, cut_out_of_image, plt_rectangles, plt_rectangles_one_col
import cv2 as cv
import sys, shutil, json
import numpy as np
from matplotlib import pyplot as plt
import os, msvcrt, copy
import pandas as pd

''' THESE ARE PARAMETERS YOU CAN CHANGE  '''

img_name = 'Original_Halved.tif'
batch_name = 'test'
figsize = 2

'''             DOWN TO HERE            '''

# This bit sets up a few initial variables:
global cnt, err_count, stats_data
cnt, prev, err_count = 0, 0, 0
img_name_no_ext = img_name.split('.')[0]

if os.path.isdir(f'imgs_classified/{batch_name}_{img_name_no_ext}'):
    inputt = str(input(f'The folder {batch_name}_{img_name_no_ext} already exists are you sure you want to continue? (y/n): ') or "y")
    if inputt=='n':
        sys.exit('User terminated as folder already exists')
    else:
        if not os.path.isdir(f'imgs_classified_png/{batch_name}_{img_name_no_ext}'):
            os.makedirs(f'imgs_classified_png/{batch_name}_{img_name_no_ext}')
        shutil.rmtree(f'imgs_classified/{batch_name}_{img_name_no_ext}/')
        os.makedirs(f'imgs_classified/{batch_name}_{img_name_no_ext}/')
        try:
            shutil.rmtree(f'imgs_classified_png/{batch_name}_{img_name_no_ext}/')
            os.makedirs(f'imgs_classified_png/{batch_name}_{img_name_no_ext}/')
        except:
            pass
else:
    os.makedirs(f'imgs_classified_png/{batch_name}_{img_name_no_ext}/')
    os.makedirs(f'imgs_classified/{batch_name}_{img_name_no_ext}/')

if os.path.isfile(f'imgs_rectangled/{batch_name}_{img_name_no_ext}/track_{batch_name}_{img_name_no_ext}.json'):
    with open(f'imgs_rectangled/{batch_name}_{img_name_no_ext}/track_{batch_name}_{img_name_no_ext}.json','r') as f:
        stats_data = json.load(f)
else:
    stats_data = {'rows_done': []}

def press(event):
    '''
    This function waits for a key to be pressed when the matplotlib window is open.
    If you press x it quits
    If you press z it undoes the previous action
    If you press any other key that is valid as a filename (i.e. any latter or any number) it will
    return the character to be used for the classification folder name
    '''
    global cnt, err_count, stats_data
    cnt = event.key
    if cnt != 'z':
        print('Classifying: ', event.key)
        plt.close()
    elif cnt == 'z':
        print('Undoing')
        stats_data['rows_done'] = stats_data['rows_done'][:-1]
    return event.key

'''
The main loop. It goes through all files in the folder given by batch_name
The if __name__ == '__main__' clause just checks that this is the script that is running (and nothing else is calling
this script by accident).
'''

if __name__ == '__main__':

    stats = pd.read_csv(f'imgs_rectangled/{batch_name}_{img_name_no_ext}/{img_name_no_ext}.txt', delimiter=' ')
    stats = stats.to_numpy()
    stats = np.array(stats.tolist())
    img = cv.imread(cv.samples.findFile(f"imgs/{img_name}",0))
    # img_rectangled = cv.imread(cv.samples.findFile(f"imgs_rectangled/{img_name}",0))

    for c, col in enumerate(stats):
        if c in stats_data['rows_done']:
            continue
        arr = cut_out_of_image(img, col)
        img_r_temp = plt_rectangles_one_col(img, col, color=(255, 0, 0))
        arr_big = cut_out_of_image(img_r_temp, col, border = 100, img_size=img_r_temp.shape)

        fig, ax = plt.subplots(figsize=(figsize, figsize)) # Creating the figure
        fig.canvas.mpl_connect('key_press_event', press) # Creates window for figure with function press waiting for key

        min, max = np.min(arr_big), np.max(arr_big)
        # Make a LUT (Look-Up Table) to translate image values
        LUT = np.zeros(256, dtype=np.uint8)
        LUT[min:max + 1] = np.linspace(start=0, stop=255, num=(max - min) + 1, endpoint=True, dtype=np.uint8)

        ax.imshow(LUT[arr_big]) # Fills the window with axes
        plt.show() # Shows window

        # If key pressed = z the function press undoes previous action and this closes that plot ready for the next image
        if cnt != 'z':
            plt.close()
        # If key pressed = x the program exits
        if cnt == 'x':
            with open(f'imgs_rectangled/{batch_name}_{img_name_no_ext}/track_{batch_name}_{img_name_no_ext}.json', 'w+') as f:
                json.dump(stats_data, f)
            sys.exit('Operation terminated by user')

        # Saves file according to classification
        if not os.path.isdir(f'imgs_classified/{batch_name}_{img_name_no_ext}/{cnt}'):
            os.makedirs(f'imgs_classified/{batch_name}_{img_name_no_ext}/{cnt}')
        if not os.path.isdir(f'imgs_classified_png/{batch_name}_{img_name_no_ext}/{cnt}'):
            os.makedirs(f'imgs_classified_png/{batch_name}_{img_name_no_ext}/{cnt}')

        np.save(f'imgs_classified/{batch_name}_{img_name_no_ext}/{cnt}/{c}.npy',arr)
        cv.imwrite(f'imgs_classified_png/{batch_name}_{img_name_no_ext}/{cnt}/{c}.png', arr)
        # Removes original image
        stats_data['rows_done'].append(c)
        # Saves previous index in case we need to undo this

with open(f'imgs_rectangled/{batch_name}_{img_name_no_ext}/track_{batch_name}_{img_name_no_ext}.json', 'w+') as f:
    json.dump(stats_data, f)

print(f'Done {stats.shape[0]} with {err_count} undoes.')
if err_count>0:
    print('If you want to fix the errors rerun the program without changing any settings.')


