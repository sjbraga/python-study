import numpy as np

def find_bmu(t, net, m):
    """
        Encontra o Best Matching Unit para um vetor t no SOM
        Retorna: (bmu, bmu_index) tupla onde bmu eh o vetor de maior dimensao BMU e bmu_index eh o indice desse vetor no SOM
    """
    #inicializa o index
    bmu_index = np.array([0,0])
    #inicia distancia minima para um numero bem grande
    min_dist = np.iinfo(np.int).max
    #anda pela matriz de pesos e procura menor distancia do vetor t
    for x in range(net.shape[0]):
        for y in range(net.shape[1]):
            #pesos atuais que estou considerando
            w = net[x, y, :].reshape(m,1) #transforma matriz em vetor 3D
            #calcula distancia euclidiana ao quadrado (evita tirar raiz)
            sq_dist = np.sum((w - t) ** 2) #soma as diferencas ao quadrado de cada valor do vetor
            if sq_dist < min_dist: #se distancia eh menor salva valor e index
                min_dist = sq_dist
                bmu_index = np.array([x,y])

    #depois de percorrer a matriz tenho a menor distancia e o index do vetor BMU
    #pega vetor dentro do net
    bmu = net[bmu_index[0], bmu_index[1], :].reshape(m,1)
    #retorna o bmu e o indice
    return (bmu, bmu_index)


def decay_radius(initial_radius, i, time_constant):
    return initial_radius * np.exp(-i / time_constant)

def decay_learning_rate(initial_learning_rate, i, n_iterations):
    return initial_learning_rate * np.exp(-i / n_iterations)

def calculate_influence(distance, radius):
    return np.exp(-distance / (2 * (radius**2)))