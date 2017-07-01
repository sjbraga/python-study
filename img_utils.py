from skimage.util import random_noise, img_as_float

def add_speckle(img, mean=0.1, var=0.5):
    sp = random_noise(img, mode='speckle', seed=None, mean=mean, var=var)
    return sp