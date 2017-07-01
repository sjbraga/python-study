import numpy as np
import scipy
from matplotlib import pyplot as plt
from skimage import filters, io, data

import img_utils

img = io.imread('img/brain.jpg')
# img = data.coins()
noiseImg = img_utils.add_speckle(img, mean=0.1, var=0.05)

finalImg = np.zeros(noiseImg.shape)

for i in range(noiseImg.shape[0]):
    for j in range(noiseImg.shape[1]):
        finalImg[i,j] = noiseImg[i,j]

median = [noiseImg[0,0]] * 9

for y in range(1, noiseImg.shape[0]-1):
    for x in range(1, noiseImg.shape[1]-1):
        median[0] = noiseImg[y-1, x-1]
        median[1] = noiseImg[y-1, x]
        median[2] = noiseImg[y-1, x+1]
        median[3] = noiseImg[y, x-1]
        median[4] = noiseImg[y, x]
        median[5] = noiseImg[y, x+1]
        median[6] = noiseImg[y+1, x-1]
        median[7] = noiseImg[y+1, x]
        median[8] = noiseImg[y+1, x+1]

        median.sort()
        finalImg[y,x] = median[4]


#img original
plt.subplot(1,3,1),plt.imshow(img,'gray')
plt.title('img original'), plt.xticks([]), plt.yticks([])

#noise img
plt.subplot(1,3,2), plt.imshow(noiseImg,'gray')
plt.title('noise image'), plt.xticks([]), plt.yticks([])

#median filter
plt.subplot(1,3,3), plt.imshow(finalImg,'gray')
plt.title('median filter'), plt.xticks([]), plt.yticks([])

plt.show()
