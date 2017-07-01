import numpy as np
from skimage import filters, io, data

def median_filter(img):
    finalImg = np.zeros(img.shape)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            finalImg[i,j] = img[i,j]

    median = [img[0,0]] * 9

    for y in range(1, img.shape[0]-1):
        for x in range(1, img.shape[1]-1):
            median[0] = img[y-1, x-1]
            median[1] = img[y-1, x]
            median[2] = img[y-1, x+1]
            median[3] = img[y, x-1]
            median[4] = img[y, x]
            median[5] = img[y, x+1]
            median[6] = img[y+1, x-1]
            median[7] = img[y+1, x]
            median[8] = img[y+1, x+1]

            median.sort()
            finalImg[y,x] = median[4]

    return finalImg