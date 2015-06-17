# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 22:29:01 2015

@author: Inez Wijnands
"""

#import numpy as np
import cv2
#from matplotlib import pyplot as plt

img = cv2.imread('test_edges/10011_right.jpeg',0)

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector()

# find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img, kp, color=(0,0,255))

# Print all default params
#print "Threshold: ", fast.getInt('threshold')
#print "nonmaxSuppression: ", fast.getBool('nonmaxSuppression')
#print "neighborhood: ", fast.getInt('type')
print "Total Keypoints with nonmaxSuppression: ", len(kp)

cv2.imwrite('test-sift/10011-right-fast_true.png',img2)

# Disable nonmaxSuppression
fast.setBool('nonmaxSuppression',0)
kp = fast.detect(img,None)

print "Total Keypoints without nonmaxSuppression: ", len(kp)

img3 = cv2.drawKeypoints(img, kp, color=(0,0,255))

cv2.imwrite('test-sift/10011-right-fast_false.png',img3)