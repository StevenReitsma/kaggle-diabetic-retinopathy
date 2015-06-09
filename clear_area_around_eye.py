# -*- coding: utf-8 -*-
"""
Created on Sun May 24 01:50:46 2015

@author: Tom
"""
from __future__ import division
import numpy as np
import cv2
import copy
import os
import util


def clear_area_around_eye(self, size = 256, image_dir = 'I:/AI_for_an_eyes/test/test/', target_dir = 'I:/AI_for_an_eyes/test/test_zonder_meuk_256/'):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    pic_list = os.listdir(image_dir)
    list_length = len(pic_list)
    
    util.update_progress(0)
    for j, image_name in enumerate(pic_list):
        
        img = cv2.imread(image_dir + image_name,1)
        height, width = img.shape[:2]
    
        helpert = height/size
    
        height = height/helpert
        width = width/helpert
        img = cv2.resize(img, (int(width),int(height)))
    
    
    
        ret,thresh = cv2.threshold(img, 10, 150, cv2.THRESH_BINARY)
    
        gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    
        cimg = copy.copy(img)
        gray = cv2.medianBlur(gray,5)
    
        #find circles about the size of image height
        circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20, param1=100,param2=20,minRadius=int(height/2.05),maxRadius=int(height/2)+int(height*0.03))
    
        if circles == None:
            #find circles larger than image height
            circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20, param1=100,param2=20,minRadius=int(height/2),maxRadius=int(height/2)+int(height*0.15))
            
        if circles == None:
            #find circles smaller than image height
            circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20, param1=100,param2=20,minRadius=int(height/2.25),maxRadius=int(height/2)-int(height*0.02))
    
        if circles != None:
            circles = np.uint16(np.around(circles))
            rad=0.0
        
        
            for i in circles[0,:]:
            
                if i[2]> rad:
                    rad = i[2]
                    circle = i
               
                
        
            circle_init = np.zeros(shape = cimg.shape, dtype = cimg.dtype)
            cv2.circle(circle_init,(circle[0],circle[1]),circle[2],(255,255,255),-1)
            cimg= cv2.bitwise_and(cimg, circle_init)
        
            cv2.imwrite(target_dir + image_name, cimg)   
        util.update_progress(j/list_length)
        
    util.update_progress(1)

if __name__ == '__main__':
    
    image_dir = 'I:/AI_for_an_eyes/test/test/'
    target_dir = 'I:/AI_for_an_eyes/test/test_zonder_meuk_256/'
    size = 256
    
    clear_area_around_eye(size = size, image_dir = image_dir, target_dir = target_dir)