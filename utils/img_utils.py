#from skimage.util import random_noise, img_as_float
import numpy as np

def add_speckle(img, mean=0.1, var=0.5):
    #sp = random_noise(img, mode='speckle', seed=None, mean=mean, var=var)
    row,col = img.shape
    gauss = np.random.randn(row,col)        
    sp = img + img * gauss

    return sp