#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

#%%
#importa o dataset -> .train .test .validation
#imagens 28x28 -> 784 pixels
#dataset treinamento com 55000 imagens
#shape dataset -> 55000 x 784
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#criando o modelo de rede neural com softmax
#x -> pixels da imagem de entrada
x = tf.placeholder(tf.float32, [None, 784]) #none dimensao de qualquer tamanho

#parametros do modelo sempre sao variaveis, podem ser modificados durante as computacoes
#criando os pesos
W = tf.Variable(tf.zeros([784, 10])) #multiplica cada entrada para produzir saidas de 10
b = tf.Variable(tf.zeros([10])) #soma nas saidas

#definindo modelo
y = tf.matmul(x, W) + b #matmul -> multiplicacao de matrizes

#placeholder das saidas corretas
y_ = tf.placeholder(tf.float32, [None, 10])

#medida do erro
#faz o logaritmo da saida y, multiplica pela resposta correta
#depois soma os valores da segunda dimensao de y (reduction_indices)
#depois faz a media sobre todos os exemplos
#versao instavel
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#versao estavel
cross_entropy = tf.reduce_mean(
tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

#minimiza erro com gradient descent
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#iniciando sessao
sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

#_ para ignorar parametros
for _ in range(1000): 
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#pega todos os exemplos que a predicao foi correta
#argmax de y pega a predicao do modelo, argmax de y_ pega a resposta correta
#(array onde tudo e 0 e a resposta e 1 vai voltar o index de 1)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#equal vai voltar array de boolean, se transformar em float da pra somar e tirar a media
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#calcula accuracy agora passando os dados de teste
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))