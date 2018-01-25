import numpy as np
import os
from PIL import Image

path = "/home/samira/mestrado/img/NORMAL/ORIGINAL/"
path_to_save = "/home/samira/mestrado/img/NORMAL/GRAYSCALE/"

def get_info(inputPath):
        im = Image.open(inputPath)
        print filename
        print im.size
        print im.mode

        print "-----------------------"

def load_image(intputPath):
    img = Image.open(intputPath).convert('L')
    img.load()
    data = np.asarray(img, dtype="int32")
    return data

def save_image(npdata, outputPath):
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
    img.save(outputPath)

if __name__ == '__main__':
    listing = os.listdir(path)

    for filename in listing:
        outPath = os.path.join(path_to_save, filename)
        inPath = os.path.join(path, filename)

        npdata = load_image(inPath)
        save_image(npdata, outPath)
        get_info(outPath)