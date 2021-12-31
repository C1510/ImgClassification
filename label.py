from lib.utils import threshold_image, get_connected_components, _find_getch
import cv2 as cv
import sys
import numpy as np
from matplotlib import pyplot as plt
import os, msvcrt

directory = os.fsencode('imgs_np')



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


        list = os.listdir(dir)  # dir is your directory path
        number_files = len(list)

    if c == 5:
        break


