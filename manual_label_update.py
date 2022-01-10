import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from lib.utils import threshold_image, get_connected_components, cut_out_of_image, plt_rectangles, plt_rectangles_one_col
import cv2 as cv
import sys, shutil, json
import numpy as np
from matplotlib import pyplot as plt
import os, msvcrt, copy
from os import listdir
from os.path import isfile, join, isdir

''' THESE ARE PARAMETERS YOU CAN CHANGE  '''

img_name = 'Original_Halved.tif'
batch_name = 'test'
username = 'mark'
figsize = 5

'''             DOWN TO HERE            '''

img_name_no_ext = img_name.split('.')[0]
main_dir = f'imgs_classified_png/{batch_name}_{img_name_no_ext}_{username}/'
classes = [f for f in listdir(main_dir) if isdir(join(main_dir, f))]

stats = pd.read_csv(f'imgs_rectangled/{batch_name}_{img_name_no_ext}/stats_{username}.txt', delimiter=' ')

for folder in classes:
    files = [f for f in listdir(main_dir+'/'+folder) if isfile(join(main_dir+'/'+folder, f))]
    files = [f.split('.png')[0] for f in files]
    for f in files:
        if stats['class'][int(f)]!=folder:
            print(f"changing {stats['class'][int(f)]} to {folder}")
            stats['class'][int(f)]=copy.deepcopy(folder)

stats.to_csv(f'imgs_rectangled/{batch_name}_{img_name_no_ext}/stats_{username}.txt', sep = ' ', index=False)
