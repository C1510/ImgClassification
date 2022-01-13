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
username = 'mark'
figsize = 5

'''             DOWN TO HERE            '''

# This bit sets up a few initial variables:
global cnt, err_count, stats_data, stats
cnt, prev, err_count = 0, 0, 0
img_name_no_ext = img_name.split('.')[0]

if os.path.isdir(f'imgs_classified_png/{batch_name}_{img_name_no_ext}_{username}'):
    shutil.rmtree(f'imgs_classified_png/{batch_name}_{img_name_no_ext}_{username}/')
    os.makedirs(f'imgs_classified_png/{batch_name}_{img_name_no_ext}_{username}/')
else:
    # If the folders don't exist, create them
    os.makedirs(f'imgs_classified_png/{batch_name}_{img_name_no_ext}_{username}/')

if os.path.isfile(f'imgs_rectangled/{batch_name}_{img_name_no_ext}/track_{username}.json'):
    with open(f'imgs_rectangled/{batch_name}_{img_name_no_ext}/track_{username}.json','r') as f:
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
    global cnt, err_count, stats_data, stats
    cnt = event.key
    try:
        if cnt != 'z':
            print('Classifying: ', event.key)
            plt.close()
        elif cnt == 'z':
            print('Undoing')
            stats.iloc[stats_data['rows_done'][-1], -1]='-1'
            stats_data['rows_done'] = stats_data['rows_done'][:-1]
        return event.key
    except:
        return '-1'


'''
The main loop. It goes through all files in the folder given by batch_name
The if __name__ == '__main__' clause just checks that this is the script that is running (and nothing else is calling
this script by accident).
'''

def save_classified_images(stats, img):
    # This function takes the stats file and an image, and saves your classifications to the
    # imgs_classified and imgs_classified_png folders.
    for c, col in stats.iterrows():
        if str(col[-1])=='-1':
            continue
        # Saves file according to classification
        if not os.path.isdir(f'imgs_classified_png/{batch_name}_{img_name_no_ext}_{username}/{col[-1]}'):
            os.makedirs(f'imgs_classified_png/{batch_name}_{img_name_no_ext}_{username}/{col[-1]}')
        cv.imwrite(f'imgs_classified_png/{batch_name}_{img_name_no_ext}_{username}/{col[-1]}/{c}.png', cut_out_of_image(img, col))
    return


if __name__ == '__main__':

    if not os.path.isfile(f'imgs_rectangled/{batch_name}_{img_name_no_ext}/stats_{username}.txt'):
        shutil.copy(f'imgs_rectangled/{batch_name}_{img_name_no_ext}/{img_name_no_ext}.txt', f'imgs_rectangled/{batch_name}_{img_name_no_ext}/stats_{username}.txt')

    stats = pd.read_csv(f'imgs_rectangled/{batch_name}_{img_name_no_ext}/stats_{username}.txt', delimiter=' ')
    # stats = stats.to_numpy()
    # stats = np.array(stats.tolist())
    stats['class']=stats['class'].astype(str)
    img = cv.imread(cv.samples.findFile(f"imgs/{img_name}",0))
    # img_rectangled = cv.imread(cv.samples.findFile(f"imgs_rectangled/{img_name}",0))

    for c, col in stats.iterrows():
        if str(col['class'])!='-1':
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
            # Saves the classified data to the stats file
            with open(f'imgs_rectangled/{batch_name}_{img_name_no_ext}/track_{username}.json', 'w+') as f:
                json.dump(stats_data, f)
            # Takes the stats data and saves images into imgs_classified and imgs_classified_png
            stats.to_csv(f'imgs_rectangled/{batch_name}_{img_name_no_ext}/stats_{username}.txt', sep = ' ', index=False)
            # np.savetxt(f'imgs_rectangled/{batch_name}_{img_name_no_ext}/stats_{username}.txt', stats,
            #            fmt='%.0f', delimiter=' ', header='left_top_x left_top_y x_length y_length vol class',
            #            comments='')
            # Takes the stats data and saves images into imgs_classified and imgs_classified_png
            save_classified_images(stats, img)
            sys.exit('Operation terminated by user')

        stats.iloc[c,-1]=str(cnt)
        # Removes original image
        stats_data['rows_done'].append(c)

# Saves the data about finished rows and the order in which they were done
with open(f'imgs_rectangled/{batch_name}_{img_name_no_ext}/track_{username}.json', 'w+') as f:
    json.dump(stats_data, f)

# Saves the classified data to the stats file
# np.savetxt(f'imgs_rectangled/{batch_name}_{img_name_no_ext}/stats_{username}.txt',
#            stats, fmt='%.0f', delimiter=' ', header='left_top_x left_top_y x_length y_length vol class',comments='')
stats.to_csv(f'imgs_rectangled/{batch_name}_{img_name_no_ext}/stats_{username}.txt', sep = ' ', index=False)
# Takes the stats data and saves images into imgs_classified and imgs_classified_png
save_classified_images(stats, img)

print(f'{username} has done {stats.shape[0]} with {err_count} undoes.')
if err_count>0:
    print('If you want to fix the errors rerun the program without changing any settings.')


