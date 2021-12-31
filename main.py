from lib.utils import threshold_image
import cv2 as cv
import sys
import numpy as np
from matplotlib import pyplot as plt


img = cv.imread(cv.samples.findFile("imgs/test1.tif",0))

a, b = threshold_image(img, 25, mode = 'mg')

n, labels, stats, centroids = cv.connectedComponentsWithStats(b)

stats = np.array(stats)
mask = np.array([True if stats[i,2]*stats[i,3]>20 else False for i in range(stats.shape[0])])
stats = stats[mask,:]

print(stats.shape)

plt.imshow(b,'gray',vmin=0,vmax=255)
plt.show()
