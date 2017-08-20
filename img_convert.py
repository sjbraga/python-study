import os
from PIL import Image

path = "/media/sf_SharedWorkspace/OCT/Normal/JPG/"
path_to_save = "/media/sf_SharedWorkspace/OCT/Normal/GRAYSCALE/"

listing = os.listdir(path)

for filename in listing:
    im = Image.open(path + filename).convert('L')
    im.save(os.path.join(path_to_save, filename))