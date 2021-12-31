from lib.utils import threshold_image, get_connected_components, plt_rectangles
import cv2 as cv
import sys
import numpy as np
from matplotlib import pyplot as plt

thresholding_level = 50
min_side = 10
max_side = 100



#img_original = cv.imread(cv.samples.findFile("imgs/Original_halved.tif",0))
img_original = cv.imread(cv.samples.findFile("imgs/test1.tif",0))
a, img = threshold_image(img_original, thresholding_level, mode = 'mg')
stats = get_connected_components(img, min_side=min_side, max_side=max_side)

for c, col in enumerate(stats):
    np.save(f'imgs_np/{c}.npy',img_original[col[1]:col[1]+col[3], col[0]:col[0]+col[2]])
    # if c<2:
    #     plt.imshow(img_original[col[1]:col[1]+col[3], col[0]:col[0]+col[2]], 'gray', vmin=0, vmax=255)
    #     plt.show()

img2 = plt_rectangles(img, stats)

print(stats.shape[0],'fossils were found')

plt.imshow(img2,'gray',vmin=0,vmax=255)
plt.savefig('save.png',dpi=1000)
plt.show()
