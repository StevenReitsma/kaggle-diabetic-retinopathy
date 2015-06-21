# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import cv2
import copy
import os
import util
import sys
import time

def worker(id, jobs, result, target_dir, size):
    while True:
        task = jobs.get() #task is name of image
        if task is None:
            print "Stopping worker " + str(id)+'\n'
            break
        process_image(task, target_dir, size)
        result.put(task)

def process_image(image_path = 'I:/AI_for_an_eyes/test/test_picca.jpeg', target_dir = 'I:/AI_for_an_eyes/test/test_zonder_meuk_256/', size = 256):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)


    image_name = image_path.split('/')[-1]


    img = cv2.imread(image_path,1)

    cimg = copy.copy(img)
    height, width = img.shape[:2]

    helpert = height/size
    small_height = height/helpert
    small_width = width/helpert
    small_img = cv2.resize(img, (int(small_width),int(small_height)))

    ret,thresh = cv2.threshold(small_img, 10, 150, cv2.THRESH_BINARY)

    gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray,5)

    #find circles about the size of image height
    circles = cv2.HoughCircles(gray,cv2.cv.CV_HOUGH_GRADIENT,1,20, param1=100,param2=20,minRadius=int(small_height/2.05),maxRadius=int(small_height/2)+int(small_height*0.03))

    if circles is None:
        #find circles larger than image height
        circles = cv2.HoughCircles(gray,cv2.cv.CV_HOUGH_GRADIENT,1,20, param1=100,param2=20,minRadius=int(small_height/2),maxRadius=int(small_height/2)+int(small_height*0.15))

    if circles is None:
        #find circles smaller than image height
        circles = cv2.HoughCircles(gray,cv2.cv.CV_HOUGH_GRADIENT,1,20, param1=100,param2=20,minRadius=int(small_height/2.25),maxRadius=int(small_height/2)-int(small_height*0.02))

    if not circles is None:
        circles = np.uint16(np.around(circles))
        rad=0.0

        for i in circles[0,:]:
            if i[2]> rad:
                rad = i[2]
                circle = i

        circle_init = np.zeros(shape = cimg.shape, dtype = cimg.dtype)
        cv2.circle(circle_init, (int((circle[0]/small_width)*width), int((circle[1]/small_height)*height)), int(circle[2]*helpert), (255,255,255), -1)
        cimg= cv2.bitwise_and(cimg, circle_init)

    cv2.imwrite(target_dir + image_name, cimg)




def clear_area_around_eye(size = 256, image_dir = 'I:/AI_for_an_eyes/test/test/', target_dir = 'I:/AI_for_an_eyes/test/test_zonder_meuk_256/'):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    util.update_progress(0)


    tasks = glob.glob(image_dir+'*.jpeg')
    job_total = len(tasks)

    print 'Processing images matching ' + image_dir+ '*.jpeg'

    jobs = Queue()
    result = JoinableQueue()
    NUMBER_OF_PROCESSES = cpu_count()*2

    for im_name in tasks:
        jobs.put(im_name)

    for i in xrange(NUMBER_OF_PROCESSES):
        p = Thread(target=worker, args=(i, jobs, result, target_dir, size))
        p.daemon = True
        p.start()

    print 'Starting workers (', NUMBER_OF_PROCESSES, ')!'

    n_complete = 0
    for t in xrange(len(tasks)):
        r = result.get()
        n_complete += 1
        util.update_progress(n_complete/job_total)
        result.task_done()
        #print t, 'done'

    for w in xrange(NUMBER_OF_PROCESSES):
        jobs.put(None)

    util.update_progress(1)

    print 'Done!'
    time.sleep(1)
    result.join()
    jobs.close()
    result.close()



if __name__ == '__main__':

    args = sys.argv

    if len(args) < 3:
        image_dir = 'I:/AI_for_an_eyes/test/test/'
        target_dir = 'I:/AI_for_an_eyes/test/test_zonder_meuk_259/'
    elif len(args) > 2:
        image_dir = args[1]
        target_dir = args[2]

    else:
        throw ("Failure in input arguments! " + args)
    size = 256

    clear_area_around_eye(size=size, image_dir=image_dir, target_dir = target_dir)
