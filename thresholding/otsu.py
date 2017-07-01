import cv2 as ocv
import numpy as np
from matplotlib import pyplot as plt

def otsu_threshold(img):
#img = ocv.imread('brain.jpg',0)
#blur = ocv.GaussianBlur(img, (5,5), 0)
    height, width = img.shape

    #gera histograma
    hist = ocv.calcHist([img], [0], None, [256], [0,256])
    print hist

    bins = np.arange(256) #valores de intensidade dos pixels

    current_max = 0
    thresh = 0

    total = height * width

    #normaliza histograma, otsu considera como distribuicao de probabilidade
    #tudo deve somar pra 1
    hist_norm = hist.ravel()/total

    #foreground - pixels claros > t
    #background - pixels escuros <= t
    sum_all, sumFore, sumBack = 0, 0, 0

    for i in range(0,256):
        sum_all += i * hist[i][0] #sum_i=1^L i * p_i

    weightBack, weightFore = 0, 0
    varBetween, meanBack, meanFore = 0, 0, 0

    for i in range(0, 256):
        weightBack += hist[i][0] #w_0 = sum_i=1^k = w(k)
        weightFore = total - weightBack #w_1 = 1 - w(k)
        if weightFore == 0:
            break
        sumBack += i*hist[i][0] #sum_i=1^k i*p_i
        sumFore = sum_all - sumBack #diferenca do que ja somei para o total
        meanBack = sumBack / weightBack #sum_i=1^k i*p_i / w(k)
        meanFore = sumFore / weightFore 
        varBetween = weightBack * weightFore * (meanBack-meanFore)**2
        if(varBetween > current_max): #max 1 < k <= L varBetween
            current_max = varBetween
            thresh = i

    print "threshold: ", thresh

    ret1, globalThresh = ocv.threshold(img, 127, 255, ocv.THRESH_BINARY)
    ret2, otsuThresh = ocv.threshold(img, thresh, 255, ocv.THRESH_BINARY)
    ret3, th2 = ocv.threshold(img, 0, 255, ocv.THRESH_BINARY+ocv.THRESH_OTSU)

    print "opencv threshold: ", ret3

    return otsuThresh

# plt.figure(figsize=(20,10))

# #img original
# plt.subplot(2,2,1),plt.imshow(img,'gray')
# plt.title('img original'), plt.xticks([]), plt.yticks([])

# #histograma
# plt.subplot(2,2,2), plt.plot(hist), plt.xlim([0,256]), plt.ylim([0, hist.max()])
# plt.title('histograma')

# plt.subplot(2,2,3), plt.imshow(globalThresh,'gray')
# plt.title('global threshold'), plt.xticks([]), plt.yticks([])

# plt.subplot(2,2,4), plt.imshow(otsuThresh,'gray')
# plt.title('otsu threshold'), plt.xticks([]), plt.yticks([])

# # plt.subplot(3,2,5), plt.imshow(th2,'gray')
# # plt.title('otsu threshold opencv'), plt.xticks([]), plt.yticks([])

# plt.show()