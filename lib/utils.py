import copy

import cv2 as cv
import sys, math
import numpy as np
from matplotlib import pyplot as plt


def threshold_image(img, thresh, mode = 'mg'):
    if mode == 'mg':
        thresh_method = cv.ADAPTIVE_THRESH_GAUSSIAN_C
    elif mode == 'bi':
        thresh_method = cv.THRESH_BINARY
    ret, thresh = cv.threshold(img, thresh, 255, thresh_method)
    thresh = cv.cvtColor(thresh, cv.COLOR_BGR2GRAY)
    return ret, thresh


def get_connected_components(img, min_side, max_side=None, border = 0):
    b = cv.bitwise_not(img)
    n, labels, stats, centroids = cv.connectedComponentsWithStats(b)

    min = min_side**2
    if max_side is not None:
        max = max_side **2

    stats = np.array(stats)
    if max_side is not None:
        mask = np.array([True if (max > stats[i, 2] * stats[i, 3] > min and stats[i, 2]>1 and stats[i, 3]>1) else False for i in range(stats.shape[0])])
    else:
        mask = np.array([True if (stats[i, 2] * stats[i, 3] > min and stats[i, 2]>1 and stats[i, 3]>1) else False for i in range(stats.shape[0])])

    stats = stats[mask, :]

    mask = [True if (np.mean(img[col[1]:col[1]+col[3], col[0]:col[0]+col[2]])>10 and not math.isnan(np.mean(img[col[1]:col[1]+col[3], col[0]:col[0]+col[2]]))) else False for col in stats]
    stats = stats[mask, :]

    img_size = b.shape

    stats[:, 0] = np.maximum(stats[:, 0]-border, 0)
    stats[:, 1] = np.maximum(stats[:, 1]-border, 0)
    stats[:, 2] = np.minimum(stats[:, 2]+2*border, img_size[1]-stats[:, 0])
    stats[:, 3] = np.minimum(stats[:, 3]+2*border, img_size[0]-stats[:, 1])

    return stats


def plt_rectangles(img, stats, color = (255,0,0)):
    try:
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    except:
        pass
    for col in stats:
        cv.rectangle(img, (col[0], col[1]), (col[0] + col[2], col[1] + col[3]), color, 1)
    return img

def plt_rectangles_one_col(img, col, color = (255,0,0)):
    img = copy.deepcopy(img)
    try:
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    except:
        pass
    cv.rectangle(img, (col[0], col[1]), (col[0] + col[2], col[1] + col[3]), color, 1)
    return img


def divide_image(img_original, stats, batch_name, mode = 'np'):
    len_stats = len(stats)
    for c, col in enumerate(stats):
        if mode == 'np':
            np.save(f'lib/imgs_np/{batch_name}/{c}.npy', cut_out_of_image(img_original, col))
        elif mode == 'png':
            cv.imwrite(f'lib/imgs_png/{batch_name}/{c}.png', cut_out_of_image(img_original, col))
    return

def cut_out_of_image(img, col, border = 0, img_size=(1e8,1e8)):
    col = copy.deepcopy(col)
    if border !=0:
        col[0] = np.maximum(col[0] - border, 0)
        col[1] = np.maximum(col[1] - border, 0)
        col[2] = np.minimum(col[2] + 2 * border, img_size[1] - col[0])
        col[3] = np.minimum(col[3] + 2 * border, img_size[0] - col[1])
    return img[col[1]:col[1] + col[3], col[0]:col[0] + col[2]]
