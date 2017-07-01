import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('dog.png', 0)

#threshold normal no nivel de cinza 127
ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

#threshold com otsu
ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#plota todas as imagens e os histogramas
images = [img, 0, th1,
            img, 0, th2]

#titulos das imagens
titles = ['img original', 'histograma', 'global threshold 127',
          'img original', 'histograma', 'otsu threshold']

for i in xrange(2):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])

plt.show()
