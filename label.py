from lib.utils import threshold_image, get_connected_components, _find_getch
import cv2 as cv
import sys
import numpy as np
from matplotlib import pyplot as plt
import os, msvcrt

batch_name = 'test'

directory = os.fsencode(f'imgs_np/{batch_name}')

if  os.path.isdir(directory):
    inputt = str(input(f'The folder {batch_name} already exists are you sure you want to continue?') or "y")
    if inputt=='n':
        sys.exit('User terminated as folder already exists')

def press(event):
    global cnt
    print('press', event.key)
    cnt = event.key
    plt.close()
    return event.key

for c, file in enumerate(os.listdir(directory)):
    filename = os.fsdecode(file)
    if filename.endswith(".npy"):
        global cnt
        cnt = 0
        fig, ax = plt.subplots()
        arr = np.load(str(directory, 'UTF8')+'/'+str(filename))
        fig.canvas.mpl_connect('key_press_event', press)
        ax.imshow(arr)
        plt.show()
        plt.close()

        if cnt == 'x':
            sys.exit('Operation terminated by user')
        if not os.path.isdir(f'imgs_classified/{batch_name}/{cnt}'):
            os.makedirs(f'imgs_classified/{batch_name}/{cnt}')
        next_num = len(os.listdir(f'imgs_classified/{batch_name}/{cnt}'))
        np.save(f'imgs_classified/{batch_name}/{cnt}/{next_num+1}.npy',arr)
        os.remove(str(directory, 'UTF8')+'/'+str(filename))


