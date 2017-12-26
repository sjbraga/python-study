import os
from os.path import basename
from PIL import Image
from PIL import ImageOps

path = "/home/samira/Dropbox/Mestrado/Gomi/Imagens OCT/glaucoma/"

listing = os.listdir(path)

for filename in listing:
    img = Image.open(path + filename)
    mirror_img = ImageOps.mirror(img)
    img_name = "{}_INV.png".format(os.path.splitext(filename)[0])
    mirror_img.save(os.path.join(path, img_name))
    img.save(os.path.join(path, filename))
