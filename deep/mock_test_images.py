import numpy as np
import scipy.misc



if __name__ == '__main__':
    size = 256
    gen_sizes = [1000, 300, 150, 200, 50]


    red_image = np.ones((3,size,size))
    red_image[0] = red_image[0]*255

    green_image = np.ones((3,size,size))
    green_image[1] = green_image[1]*255

    yellow_image = np.ones((3,size,size))
    yellow_image[1] = yellow_image[1]*255
    yellow_image[0] = yellow_image[0]*255

    blue_image = np.ones((3,size,size))
    blue_image[2] = blue_image[2]*255

    black_image = np.ones((3,size,size))

    images = [red_image, green_image, yellow_image, blue_image, black_image]


    labels = {}

    for c, count in enumerate(gen_sizes):
        for i, count in enumerate(range(count)):
            image_name = str(c)+'-'+str(i)
            path = '../data/processed_mock/train/'+image_name+'.jpeg'
            labels[image_name] = str(c)
            scipy.misc.imsave(path, images[c])

    print labels

    with open("../data/trainLabels.csv", "w") as text_file:
        text_file.write("image,level")
        for image, label in labels.iteritems():
            text_file.write('\n'+image +','+ label)
