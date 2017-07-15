import numpy as np
from matplotlib import pyplot as plt
from skimage import io, data

import filters as ft
from utils import img_utils

#img = io.imread('img/woman_gs.jpg')
#img = data.coins()
#noiseImg = io.imread('img/woman.png', True)
#noiseImg = img_utils.add_speckle(img, mean=0, var=0.05)

img = np.random.normal(0.5, 0.1, (100,100))
img[:,:50] += 0.25

finalImg = ft.lee_filter(img, 3)
        


#img original
plt.subplot(2,2,1),plt.imshow(img,'gray')
plt.title('img original'), plt.xticks([]), plt.yticks([])

#noise img
# plt.subplot(2,2,2), plt.imshow(noiseImg,'gray')
# plt.title('noise image'), plt.xticks([]), plt.yticks([])

#filter
plt.subplot(2,2,3), plt.imshow(finalImg,'gray')
plt.title('filter'), plt.xticks([]), plt.yticks([])

#filter
# plt.subplot(2,2,4), plt.imshow(compareImag,'gray')
# plt.title('compare'), plt.xticks([]), plt.yticks([])

plt.show()