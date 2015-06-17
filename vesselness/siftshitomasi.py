# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 05:10:17 2015

@author: Inez Wijnands
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('test_edges/10011_right-badcrop.jpeg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray,250,0.01,10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)

plt.imshow(img),plt.show()
cv2.imwrite('test-sift/siftshitomasi10011right-badcrop.jpeg',img)