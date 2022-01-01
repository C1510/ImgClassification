from lib.utils import threshold_image, get_connected_components
import cv2 as cv
import sys
import numpy as np
from matplotlib import pyplot as plt
import os, msvcrt, copy

''' THESE ARE PARAMETERS YOU CAN CHANGE  '''

batch_name = 'test'

'''             DOWN TO HERE            '''

# This bit sets up a few initial variables:
directory = os.fsencode(f'imgs_np/{batch_name}')
global cnt, next_num, cnt_prev, in_dir_files, err_count
cnt, next_num, prev, err_count = 0, 0, 0, 0
in_dir_files = os.listdir(directory)
if os.path.isdir(f'imgs_classified/{batch_name}'):
    inputt = str(input(f'The folder {batch_name} already exists are you sure you want to continue? (y/n): ') or "y")
    if inputt=='n':
        sys.exit('User terminated as folder already exists')


def press(event):
    '''
    This function waits for a key to be pressed when the matplotlib window is open.
    If you press x it quits
    If you press z it undoes the previous action
    If you press any other key that is valid as a filename (i.e. any latter or any number) it will
    return the character to be used for the classification folder name
    '''
    global cnt, next_num, cnt_prev, err_count
    print('press', event.key)
    cnt = event.key
    if cnt != 'z':
        plt.close()
    elif cnt == 'z':
        try:
            err_count += 1
            err_num = len([i for i in os.listdir(f'imgs_np/{batch_name}/') if ('err' in i)])
            os.rename(f'imgs_classified/{batch_name}/{cnt_prev}/{next_num + 1}.npy',
                      f'imgs_np/{batch_name}/err_{err_num + 1}.npy')
            print('moving and undoing')
        except:
            print('You have already done one undo, try a different key (sorry only one undo implemented)')

    return event.key

'''
The main loop. It goes through all files in the folder given by batch_name
The if __name__ == '__main__' clause just checks that this is the script that is running (and nothing else is calling
this script by accident).
'''

if __name__ == '__main__':

    for c, file in enumerate(in_dir_files):
        filename = os.fsdecode(file)
        # This if statement checks that the file is a numpy array (which is how I save the fossil examples)
        if filename.endswith(".npy"):
            cnt = 0
            fig, ax = plt.subplots() # Creating the figure
            arr = np.load(str(directory, 'UTF8')+'/'+str(filename)) # Opening the numpy array
            fig.canvas.mpl_connect('key_press_event', press) # Creates window for figure with function press waiting for key
            ax.imshow(arr) # Fills the window with axes
            plt.show() # Shows window

            # If key pressed = z the function press undoes previous action and this closes that plot ready for the next image
            if cnt != 'z':
                plt.close()
            # If key pressed = x the program exits
            if cnt == 'x':
                sys.exit('Operation terminated by user')

            # Saves file according to classification
            if not os.path.isdir(f'imgs_classified/{batch_name}/{cnt}'):
                os.makedirs(f'imgs_classified/{batch_name}/{cnt}')
            next_num = len(os.listdir(f'imgs_classified/{batch_name}/{cnt}'))
            np.save(f'imgs_classified/{batch_name}/{cnt}/{next_num+1}.npy',arr)
            # Removes original image
            os.remove(str(directory, 'UTF8')+'/'+str(filename))
            # Saves previous index in case we need to undo this
            cnt_prev = copy.deepcopy(cnt)
3
print(f'Done {len(in_dir_files)} with {err_count} undoes.')
if err_count>0:
    print('If you want to fix the errors rerun the program without changing any settings.')


