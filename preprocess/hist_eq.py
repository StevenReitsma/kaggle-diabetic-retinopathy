# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import cv2
import os
import util
import glob


def hist_eq(image_dir = 'test_hist/', target_dir = 'test_result_hist/', method = 'CLAHE'):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    #pic_list = os.listdir(image_dir)
    pic_list = glob.glob(image_dir+'/*.jpeg')
    list_length = len(pic_list)
    
    util.update_progress(0)
    for j, image_path in enumerate(pic_list):
        
        img = cv2.imread(image_path,1)
        # Use file name only, without .jpeg
        image_name = image_path.split('/')[-1][:-5] 
        
        b,g,r = cv2.split(img)        
        
        if method == 'HE':
            cv2.equalizeHist(b,b)
            cv2.equalizeHist(g,g)
            cv2.equalizeHist(r,r)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            clahe.apply(g,g)
            if not method =='CLAHE_G':
                clahe.apply(b,b)
                clahe.apply(r,r)
            
        recombined = cv2.merge((b,g,r))
        cv2.imwrite(target_dir + image_name + method +'.jpeg', recombined)
        util.update_progress(j/list_length)
        
    util.update_progress(1)

if __name__ == '__main__':
    
	if len(args) < 4:
		image_dir = 'test_hist/'
		target_dir = 'test_result_hist/'
		method = 'CLAHE_G' 
		print "Using default params!"
	elif len(args) > 3:
		image_dir = args[1]
		target_dir = args[2]
		method = args[3].upper()
	else:
		except "Failure in input arguments!", args
    
    #'CLAHE' for adaptive 
    #'CLAHE_G' only green channe
    #'HE' for normal hist equalization
    
    hist_eq(image_dir=image_dir, target_dir = target_dir, method = method)