from __future__ import print_function

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

for img_no in xrange(1, 323):
# def show(img_no=1):
    img = cv.imread('enhanced/{}.png'.format(img_no), cv.IMREAD_GRAYSCALE)
    img_blur = cv.blur(img, (5,5))
    # threshold = threshold_otsu(img, nbins=255)
    threshold = 23
    binary = img_blur > threshold
    idx = np.argwhere(binary==True)
    result = np.zeros(img.shape)
    for [x,y] in idx:
        result[x, y] = img[x, y]
    cv.imwrite('segmented/{}.png'.format(img_no), result)
    print('Done.. {}'.format(img_no))