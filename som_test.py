import numpy as np
from matplotlib import patches as patches
from matplotlib import pyplot as plt

from SOM import som

#Objetivo: agrupar as cores RGB em formato 2D juntando as cores similares na mesma regiao
#Input: matriz 100x3 (100 linhas de vetor 3D com valores RGB das cores) = 100 cores aleatorias
#Output: mapeamento em matriz 5x5 de vetores 3D com valores RGB das cores geradas pelos pesos do SOM
#Ao final gera dois graficos mostrando input inicial e output do SOM

#cria dataset de cores RGB
raw_data = np.random.randint(0, 255, (3,100)) #100 linhas de vetor 3D valores 0 a 255

#criando dimensoes do SOM -> grid 5x5
network_dimensions = np.array([5,5])
#inicializando parametros do SOM
n_iterations = 4000
init_learning_rate = 0.1

#treino do SOM, retorna rede SOM 5x5 com as cores definidas pelos pesos
net = som.training(data_to_train=raw_data, network_dimensions=network_dimensions, n_iterations=n_iterations, init_learning_rate=init_learning_rate)

#normalizando raw_data para mostrar no grafico
data = np.float64(raw_data.reshape(10,10,3))
data = data / data.max()

fig = plt.figure()

ax = fig.add_subplot(111, aspect='equal')
ax.set_xlim((0, net.shape[0]+1))
ax.set_ylim((0, net.shape[1]+1))
ax.set_title('Self-Organising Map after %d iterations' % n_iterations)

for x in range(1, net.shape[0] + 1):
    for y in range(1, net.shape[1] + 1):
        ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1,
                     facecolor=net[x-1,y-1,:],
                     edgecolor='none'))
                     
fig2 = plt.figure()

ax_in = fig2.add_subplot(111, aspect='equal')
ax_in.set_xlim((0, data.shape[0]+1))
ax_in.set_ylim((0, data.shape[1]+1))
ax_in.set_title('Input matrix do SOM')

for x in range(1, data.shape[0] + 1):
    for y in range(1, data.shape[1] + 1):
        ax_in.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1,
                     facecolor=data[x-1,y-1,:],
                     edgecolor='none'))


plt.show()
