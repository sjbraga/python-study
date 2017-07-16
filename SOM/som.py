import numpy as np

def training(data_to_train, network_dimensions=np.array([5,5]), n_iterations=2000, init_learning_rate=0.01, normalise_data=True, normalise_by_column=False):
    """
        Processo de treinamento da rede Kohonen Self Organizing Map
        http://blog.yhat.com/posts/self-organizing-maps-2.html
        data_to_train: dados de treinamento (input vector)
        network_dimensions: tamanho do SOM - geralmente menor do que o input vector
        n_iterations: numero de iteracoes de treinamento
        init_learning_rate: taxa de aprendizado inicial
        normalise_data: manter valores entre 0 e 1
        normalise_by_column: se cada coluna tem ordem de grandeza diferente, normalizacao precisa ser feita com valores de cada coluna
        Retorna: matriz de saida com o "Mapa", matriz do tamanho network_dimensions com os pesos de cada neuronio
    """
    #transforma em float para fazer normalizacao
    raw_data = np.float64(data_to_train)

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

    #cria matriz auxiliar caso precise normalizar
    data = raw_data

    if normalise_data:
        data = normalise(raw_data, normalise_by_column)
    
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
        bmu, bmu_index = find_bmu(t, net, m)

        #diminui parametros de aprendizado usando
        #usa exponetial decay sigma_t = sigma_0 * exp(-t / lambda)
        #sigma_t eh o novo valor
        #sigma_0 eh o valor anterior
        #t eh o instante de tempo
        #lamba eh o time_constant
        r = decay_radius(init_radius, i, time_constant)
        l = decay_learning_rate(init_learning_rate, i, n_iterations)

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
                    influence = calculate_influence(w_dist, r)
                    #atualiza pesos do neuronio
                    #w_novo = w_atual + (aprendizado * influencia * delta)
                    #delta = entrada - w_atual
                    new_w = w + (l * influence * (t - w))
                    #coloca novo peso na matriz
                    net[x, y, :] = new_w.reshape(1, 3)

    
    return net


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
    """
        Diminui o raio usando exponential decay baseado em uma constante time constant
    """
    return initial_radius * np.exp(-i / time_constant)

def decay_learning_rate(initial_learning_rate, i, n_iterations):
    """
        Diminui a taxa de aprendizado usando exponential decay baseada no numero de interacoes total
    """
    return initial_learning_rate * np.exp(-i / n_iterations)

def calculate_influence(distance, radius):
    """
        Calcula o grau de influencia baseada na distancia entre os dois neuronios e o raio considerado
    """
    return np.exp(-distance / (2 * (radius**2)))

def normalise(raw_data, normalise_by_column=False):
    """
        Normaliza dados para que uma feature nao seja mais importante que as outras
        Retorna: dados normalizados
    """
    data = raw_data
    if normalise_by_column:
        #normaliza valores usando o maximo de cada coluna
        col_maxes = raw_data.max(axis=0)
        #divide cada valor pelo maximo correspondente de cada coluna
        data = raw_data / col_maxes[np.newaxis,:] 
    else:
        #divide todos os valores pelo maximo do dataset (tudo na mesma escala)
        data = raw_data / raw_data.max()

    return data
