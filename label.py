from lib.utils import threshold_image, get_connected_components, _find_getch
import cv2 as cv
import sys
import numpy as np
from matplotlib import pyplot as plt
import os, msvcrt, copy

batch_name = 'test'
directory = os.fsencode(f'imgs_np/{batch_name}')

global cnt, next_num, cnt_prev, in_dir_files
cnt, next_num, prev = 0, 0, 0
in_dir_files = os.listdir(directory)

if  os.path.isdir(directory):
    inputt = str(input(f'The folder {batch_name} already exists are you sure you want to continue?') or "y")
    if inputt=='n':
        sys.exit('User terminated as folder already exists')

def press(event):
    global cnt, next_num, cnt_prev
    print('press', event.key)
    cnt = event.key
    if cnt != 'z':
        plt.close()
    elif cnt == 'z':
        print('moving and undoing')
        err_num = len([i for i in os.listdir(f'imgs_classified/{batch_name}/') if ('err' in i)])
        os.rename(f'imgs_classified/{batch_name}/{cnt_prev}/{next_num + 1}.npy',
                  f'imgs_np/{batch_name}/err_{err_num + 1}.npy')

    return event.key

for c, file in enumerate(in_dir_files):
    filename = os.fsdecode(file)
    if filename.endswith(".npy"):
        cnt = 0
        fig, ax = plt.subplots()
        arr = np.load(str(directory, 'UTF8')+'/'+str(filename))
        fig.canvas.mpl_connect('key_press_event', press)
        ax.imshow(arr)
        plt.show()
        if cnt != 'z':
            plt.close()

        if cnt == 'x':
            sys.exit('Operation terminated by user')

        if not os.path.isdir(f'imgs_classified/{batch_name}/{cnt}'):
            os.makedirs(f'imgs_classified/{batch_name}/{cnt}')
        next_num = len(os.listdir(f'imgs_classified/{batch_name}/{cnt}'))
        np.save(f'imgs_classified/{batch_name}/{cnt}/{next_num+1}.npy',arr)
        os.remove(str(directory, 'UTF8')+'/'+str(filename))
        cnt_prev = copy.deepcopy(cnt)


