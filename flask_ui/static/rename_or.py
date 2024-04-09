import os
import cv2 as cv

for image in xrange(1,323):
    img = cv.imread('original/mdb{:03}.pgm'.format(image))
    cv.imwrite('original/{}.png'.format(image), img)
    print "Done.. {}".format(image)