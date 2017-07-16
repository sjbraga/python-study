import numpy as np
from matplotlib import patches as patches
from matplotlib import pyplot as plt

from SOM import som

#Objetivo: agrupar as cores RGB em formato 2D juntando as cores similares na mesma regiao

#cria dataset de cores RGB
raw_data = np.float64(np.random.randint(0, 255, (3,100))) #100 linhas de vetor 3D valores 0 e 255

#criando SOM -> grid 5x5
network_dimensions = np.array([10,10])
#inicializando parametros do SOM
n_iterations = 4000
init_learning_rate = 0.01
#tamanho baseado nos dados
m = raw_data.shape[0]
n = raw_data.shape[1]

#matriz de pesos tem que ter o mesmo tamanho do vetor de entrada (RGB = 3 entradas) 
#para cada neuronio no mapa (mapa 5x5)
#inicializa pesos com valores aleatorios
net = np.random.random((network_dimensions[0], network_dimensions[1], m))

#raio da vizinhanca inicial (qual distancia eu procuro por vizinhos para atualizar)
init_radius = max(network_dimensions[0], network_dimensions[1]) / 2
#quanto o raio ira diminuir
time_constant = n_iterations / np.log(init_radius)

normalise_data = True
normalise_by_column = False

data = np.zeros(raw_data.shape)

if normalise_data:
    if normalise_by_column:
        #normaliza valores usando o maximo de cada coluna
        col_maxes = raw_data.max(axis=0)
        #divide cada valor pelo maximo correspondente de cada coluna
        data = raw_data / col_maxes[np.newaxis,:] 
    else:
        #divide todos os valores pelo maximo do dataset (tudo na mesma escala)
        data = raw_data / raw_data.max()

#PROCESSO DE APRENDIZADO:
#1. Encontra o neuronio com o vetor 3D mais proximo do vetor 3D do dataset - Best Matching Unit
#
#2. Move o vetor do neuronio BMU mais proximo do vetor de entrada no espaco
#
#3. Identifica os neuronios vizinhos do BMU e move os vetores mais proximos
#
#4. Reduz taxa de aprendizado
for i in range(n_iterations):
    #seleciona um exemplo aleatorio do dataset
    t = data[:, np.random.randint(0,n)].reshape(np.array([m, 1]))

    #encotra o Best Matching Unit
    bmu, bmu_index = som.find_bmu(t, net, m)

    #diminui parametros de aprendizado usando
    #usa exponetial decay sigma_t = sigma_0 * exp(-t / lambda)
    #sigma_t eh o novo valor
    #sigma_0 eh o valor anterior
    #t eh o instante de tempo
    #lamba eh o time_constant
    r = som.decay_radius(init_radius, i, time_constant)
    l = som.decay_learning_rate(init_learning_rate, i, n_iterations)

    #move o BMU e seus vizinhos mais perto
    #atualizando pesos do BMU: w_t+1 = w_t + L_t * (V_i - w_t)
    #peso atual mais diferenca entre vetor de entrada e peso atual multipicado pela taxa de aprendiz
    #movendo o BMU mais perto do vetor de entrada
    #
    #depois, encontra outros neuronios dentro do raio definido
    #atualiza peso desses neuronios proporcionalmente a distancia ate o BMU (gaussiana)
    #para calcular essa influencia usa i_t = exp(-d^2 / (2 * sigma^2_t))
    #onde d eh a distancia entre os neuronios e sigma eh o raio no tempo atual

    for x in range(net.shape[0]):
        for y in range(net.shape[1]):
            w = net[x, y, :].reshape(m, 1) #pesos do neuronio atual
            #pega distancia euclidiana quadrada entre
            #posicao do neuronio atual e indice do bmu
            w_dist = np.sum((np.array([x, y]) - bmu_index) ** 2)
            #se a distancia eh menor que o raio atual (ao quadrado pq a distancia eh quadrada)
            if w_dist <= r**2:
                #calcula influencia do neuronio
                influence = som.calculate_influence(w_dist, r)
                #atualiza pesos do neuronio
                #w_novo = w_atual + (aprendizado * influencia * delta)
                #delta = entrada - w_atual
                new_w = w + (l * influence * (t - w))
                #coloca novo peso na matriz
                net[x, y, :] = new_w.reshape(1, 3)

fig = plt.figure()
# setup axes
ax = fig.add_subplot(111, aspect='equal')
ax.set_xlim((0, net.shape[0]+1))
ax.set_ylim((0, net.shape[1]+1))
ax.set_title('Self-Organising Map after %d iterations' % n_iterations)

# plot the rectangles
for x in range(1, net.shape[0] + 1):
    for y in range(1, net.shape[1] + 1):
        ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1,
                     facecolor=net[x-1,y-1,:],
                     edgecolor='none'))
plt.show()
