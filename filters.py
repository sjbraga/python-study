import numpy as np
from skimage import filters, io, data

COEF_VAR_DEFAULT = 0.01
Q_HOMOGENEOUS = 0.25

def lee_filter(image, window_size):
    finalImg = np.zeros(image.shape)
    img = np.float64(image)
    win_offset = window_size / 2

    N, M = img.shape

    image_std = img.std()
    image_mean = img.mean()

    Q_HOMOGENEOUS = image_std / image_mean

    for i in xrange(0,N):
        xleft = i - win_offset
        xright = i + win_offset

        if(xleft < 0):
            xleft = 0
        if(xright > N):
            xright = N
        
        xright_slice = xright
        if xright_slice < N:
            xright_slice = xright_slice + 1

        for j in xrange(0,M):
            yup = j - win_offset
            ydown = j + win_offset

            if(yup < 0):
                yup = 0
            if(ydown > M):
                ydown = M

            ydown_slice = ydown
            if ydown_slice < M:
                ydown_slice = ydown_slice + 1

            pixel_value = img[i, j] #f(x,y)
            window = img[xleft:xright, yup:ydown] #W(x,y)
            #print window
            alpha = lee_weight(window) #alpha
            #print alpha
            window_mean = window.mean() #f^barra(x,y) media de f(x,y) e seus vizinhos = janela
            #print window_mean
            new_pixel_value = (alpha * pixel_value) + ((1.0 - alpha) * window_mean)

            finalImg[i,j] = round(new_pixel_value)
    
    return finalImg


def lee_weight(window, q_hom=Q_HOMOGENEOUS):
    q2_hom = q_hom * q_hom #q^2_H

    window_mean = window.mean()
    window_std = window.std()

    q = window_std / window_mean #q(x,y)

    q2 = q * q #q(x,y)^2

    if not q2:
        q2 = COEF_VAR_DEFAULT

    a = 1.0 - (q2_hom / q2)
    
    if a < 0.0:
        a = 0.0
    else:
        a = 1.0

    return a


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


def gaussian_filter(shape=(3,3), sigma=0.5):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h