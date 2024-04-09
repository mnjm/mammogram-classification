from __future__ import print_function

import numpy as np
import cv2 as cv
# from matplotlib import pyplot as plt

# def display_2_stacked_img(img1, img2, img3, img4):
#     plt.subplot(221), plt.imshow(img1, "gray")
#     plt.subplot(222), plt.imshow(img2, "gray")
#     plt.subplot(223), plt.imshow(img3, "gray")
#     plt.subplot(224), plt.imshow(img4, "gray")
#     # figManager = plt.get_current_fig_manager()
#     # figManager.window.showMaximized()
#     plt.show()

# img_name = 1
# img = cv.imread('dataset/{}.pgm'.format(img_name), cv.IMREAD_GRAYSCALE)
# # 90 degree
# img_90 = cv.flip(img, 1, 1)
# img_180 = cv.flip(img, 0, 1)
# img_270 = cv.flip(img, -1, 1)

# display_2_stacked_img(img, img_90, img_180, img_270)
for img_no in xrange(1,323):
    img = cv.imread("dataset/{}.pgm".format(img_no), cv.IMREAD_GRAYSCALE)
    img_90 = cv.flip(img, 1, 1)
    img_180 = cv.flip(img, 0, 1)
    img_270 = cv.flip(img, -1, 1)
    cv.imwrite("dataset/{}_1.pgm".format(img_no), img_90)
    cv.imwrite("dataset/{}_2.pgm".format(img_no), img_180)
    cv.imwrite("dataset/{}_3.pgm".format(img_no), img_270)
    print("Done..{}".format(img_no))