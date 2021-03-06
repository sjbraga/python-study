#from skimage.util import random_noise, img_as_float
import numpy as np
from PIL import Image
import os

def add_speckle(img, mean=0.1, var=0.5):
    #sp = random_noise(img, mode='speckle', seed=None, mean=mean, var=var)
    row,col = img.shape
    gauss = np.random.randn(row,col)        
    sp = img + img * gauss

    return sp

def to_grayscale(path, filename, to_save_path):
    im = Image.open(path + filename).convert('L')
    im.save(os.path.join(to_save_path, filename))