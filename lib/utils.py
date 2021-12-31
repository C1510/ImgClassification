import cv2 as cv
import sys


def threshold_image(img, thresh, mode = 'mg'):
    if mode == 'mg':
        thresh_method = cv.ADAPTIVE_THRESH_GAUSSIAN_C
    elif mode == 'bi':
        thresh_method = cv.THRESH_BINARY
    ret, thresh = cv.threshold(img, thresh, 255, thresh_method)
    thresh = cv.cvtColor(thresh, cv.COLOR_BGR2GRAY)
    return ret, thresh
