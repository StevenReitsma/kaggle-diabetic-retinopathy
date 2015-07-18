from __future__ import division
import numpy as np
import cv2
import os
import util
import glob
import sys

from multiprocessing import Process, Queue, JoinableQueue, cpu_count
from threading import Thread


def worker(id, jobs, result, target_dir, method):
    while True:
        task = jobs.get() #task is name of image
        if task is None:
            print "Stopping worker " + str(id)+'\n'
            break
        process_image(task, target_dir, method)
        result.put(task)

def process_image(image_path, target_dir, method = 'CLAHE'):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    img = cv2.imread(image_path,1)
    # Use file name only, without .jpeg
    image_name = image_path.split('/')[-1].split('\\')[-1][:-5]

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

    cv2.imwrite(target_dir + image_name + '.jpeg', recombined, [cv2.cv.CV_IMWRITE_JPEG_QUALITY,100])


def hist_eq(image_dir = 'test_hist/', target_dir = 'test_result_hist/', method = 'CLAHE'):

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)


    tasks = glob.glob(image_dir+'*.jpeg')
    job_total = len(tasks)

    print 'Processing images matching ' + image_dir+ '*.jpeg'

    jobs = Queue()
    result = JoinableQueue()
    NUMBER_OF_PROCESSES = cpu_count()*2

    for im_name in tasks:
        jobs.put(im_name)

    for i in xrange(NUMBER_OF_PROCESSES):
        p = Thread(target=worker, args=(i, jobs, result, target_dir, method))
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

    print 'Done!'
    result.join()
    jobs.close()
    result.close()

if __name__ == '__main__':
    args = sys.argv

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
        throw ("Failure in input arguments! " + args)

    #'CLAHE' for adaptive
    #'CLAHE_G' only green channe
    #'HE' for normal hist equalization

    hist_eq(image_dir=image_dir, target_dir = target_dir, method = method)
