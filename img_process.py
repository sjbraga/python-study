import numpy as np
from matplotlib import pyplot as plt
from skimage import io, data

import filters
from utils import img_utils

#img = io.imread('img/brain.jpg')
img = data.coins()
noiseImg = img_utils.add_speckle(img, mean=0, var=0.5)

finalImg = filters.lee_filter(noiseImg, 3)
        


#img original
plt.subplot(1,3,1),plt.imshow(img,'gray')
plt.title('img original'), plt.xticks([]), plt.yticks([])

#noise img
plt.subplot(1,3,2), plt.imshow(noiseImg,'gray')
plt.title('noise image'), plt.xticks([]), plt.yticks([])

#filter
plt.subplot(1,3,3), plt.imshow(finalImg,'gray')
plt.title('filter'), plt.xticks([]), plt.yticks([])

plt.show()