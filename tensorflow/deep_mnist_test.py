# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#funcoes para criacao da rede convolucional --------------------

def weight_variable(shape):
    """
    Inicia pesos com um pouco de ruido para garantir que nao tenha gradiente 0
    Cria e retorna variable com valor baixo inicial
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """
    Cria neuronio de bias com valor positivo para evitar "neuronios mortos"
    Cria e retorna variable com valor inicial 0.1
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    """
    Convolucao 2D entre x e W (multipicacao de matriz)
    stride: o quanto a mascara vai andar -> 1 unidade para o lado -> diminui o tamanho da saida
    padding: coloca borda com 0 para manter a saida do mesmo tamanho da entrada
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """
    Faz a convolucao diminuindo o tamanho da saida, retorna matriz com o maximo de cada regiao do filtro
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#-------------------------------------------------

#importa o dataset -> .train .test .validation
#imagens 28x28 -> 784 pixels
#dataset treinamento com 55000 imagens
#shape dataset -> 55000 x 784
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#criando o modelo de rede neural com softmax
#x -> pixels da imagem de entrada
x = tf.placeholder(tf.float32, [None, 784]) #none dimensao de qualquer tamanho
#placeholder das saidas corretas
y_ = tf.placeholder(tf.float32, [None, 10])


#primeira camada -------------------------------
#convolucao -> ReLU -> max pooling
#cria matriz de pesos para fazer a convolucao, filtro 5x5 em um canal (greyscale) com saida 32
W_conv1 = weight_variable([5, 5, 1, 32])
#cria matriz dos bias que vao somar com todas as 32 saidas
b_conv1 = bias_variable([32])

#para aplicar na camada, faz o reshape da entrada para tensor 4D
#dimensao 2 largura da imagem dimensao 3 altura da imagem, dimensao 4 numero de canais (ex: 1 greyscale, 3 rgb)
x_image = tf.reshape(x, [-1, 28, 28, 1])

#faz a convolucao da imagem com os pesos e adiciona o bias -> faz o ReLU com a saida
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#faz o maxpooling da convolucao -> saida vira 14x14
h_pool1 = max_pool_2x2(h_conv1)


#segunda camada ---------------------------------
#camada igual a primeira mas com 64 saidas
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#maxpooling reduzindo imagem para 7x7
h_pool2 = max_pool_2x2(h_conv2)


#camada totalmente conectada --------------------
#cria camada com 1024 neuronios pra processar a imagem toda
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

#transforma saida do pool em vetor
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
#multiplica o ultimo pool pelos pesos da camada conectada e soma os pesos da camada conectada depois faz relu de tudo
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#para reduzir overfitting, cria probabilidade da saida de um neuronio ser mantida
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#camada de saida --------------------------------
#pesos para camada de saida, camada anterior tinha 1024, output das 10 saidas (digitos)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

#inclui probabilidade de manter a saida
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


#avaliando modelo ------------------------------
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d: training accuracy: %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        #salvar modelo
        saver = tf.train.Saver()
        saver.save(sess, "/home/samira/mestrado/disciplinas/ptc/python/python-study/tensorflow/model/test_deep")

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    
    