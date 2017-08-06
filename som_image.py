import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.io as io
import filters

from SOM import som

global_seg = 1.5

img = io.imread('img/brain.bmp', as_grey=True)

#vetorizando imagem para passar para o SOM
image = img.reshape(1, img.shape[0]*img.shape[1])

network_dimensions = np.array([5,5])
n_iterations = 20000
init_learning_rate = 0.1

net = som.training(data_to_train=image, network_dimensions=network_dimensions, 
            n_iterations=n_iterations, init_learning_rate=init_learning_rate)

#plt.imshow(net, 'gray')
print(net)

net_values = net.reshape(network_dimensions[0], network_dimensions[1])

# kernel_size = 2 * round(2 * global_seg) + 1

# a1, a2 = img.shape

# phi = np.ones((a1,a2))

# r = 15

# phi[r:a1-r, r:a2-r] = -1

# u = -phi

# G = filters.gaussian_filter(kernel_size, global_seg)

# ul = np.zeros((a1,a2))

# for n in range(1,1000):
#     if n > 1 && np.allclose(ul, u):
#         break

#     ul = u


fig = plt.figure()
ax = fig.add_subplot(122, aspect='equal')
ax.set_title('SOM plot')
im = ax.pcolormesh(net_values, cmap=cm.gray, edgecolors='none')

ax.patch.set(hatch='xx', edgecolor='black')

ax = fig.add_subplot(121, aspect='equal')
ax.set_title('Original img input')
imgplot = plt.imshow(img, 'gray')

plt.show()