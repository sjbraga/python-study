#Tensorflow Hello World file
#%%
from __future__ import print_function

import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#%%
#criando tensores de constante
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
#print(node1, node2)

#criando soma
node3 = tf.add(node1, node2)

#cria a sessao que avalia os tensores
sess = tf.Session()

#para avaliar os tensores criados
print('run node 3', sess.run(node3))

#%%
#placeholder: variaveis
#cria dois parametros de entrada e define a operação
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
#shortcut da soma tf.add
adder_node = a + b

#roda a sessao passando os valores para as variaiveis
print(sess.run(adder_node, {a: 4, b:7.5}))
#passando valores como vetores
print(sess.run(adder_node, {a:[1,3], b:[2,4]}))

#%%
#coloca mais uma operação depois da soma
add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a: 4, b: 9}))

#%%
#criando variaveis com valores iniciais
#variaveis nao sao inicializadas automaticamente
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)

#cria um modelo linear
linear_model = W*x + b

#inicializa variaveis
init = tf.global_variables_initializer()
sess.run(init)

#executa linear model para varios valores de x
print(sess.run(linear_model, {x: [1,2,3,5]}))

#%%
#para avaliar o desempenho do modelo, precisa criar uma função de perda
#criando placeholder de y = saidas desejadas
y = tf.placeholder(tf.float32)

#funçao de perda root mean squared
#linear_model - y cria vetor com as diferenças da saida
# com o valor esperado e eleva ao quadrado
squared_deltas = tf.square(linear_model - y)
#função de perda
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

#%%
#coloca novos valores em variaveis
fixW = tf.assign(W, [-1.])
fixB = tf.assign(b, [1.])
sess.run([fixW, fixB])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

#%%
#tf.train
#minimização do erro usando otimizador -> gradient descent
#faz o ajuste das variaveis W e b de acordo com o tensor loss
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

#reinicia as variaveis
sess.run(init)
#faz as iterações de treinamento para minimizar erro
for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([W, b]))

#obtendo resultados finais
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print("W: %s b: %s, loss: %s"%(curr_W, curr_b, curr_loss))

#%%
#tf.estimator
#mais alto nivel permite treinamento, avaliação e
#manipulação de datasets
import numpy as np

#declara as colunas de features
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

#usa estimator para chamar o training e evaluation
#estimator de regressão linear
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

#cria datasets de treinamento e validação
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])

x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

#criando os datasets, informa numero de epocas e tamanho do dataset
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

#inicia treinamento passando qtde de vezes
estimator.train(input_fn=input_fn, steps=1000)

#avaliação do modelo
#testa com os dados de treino
train_metrics = estimator.evaluate(input_fn=train_input_fn)
#testa com os dados de teste
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)

#%%
#usando estimator com modelo custom
def model_fn(features, labels, mode):
    """
    Custom model for tensorflow
    """
    #criando modelo linear para fazer predições
    W = tf.get_variable("W", [1], dtype=tf.float64)
    b = tf.get_variable("b", [1], dtype=tf.float64)
    y = W*features['x'] + b

    #calculo de erro
    loss = tf.reduce_sum(tf.square(y - labels))

    #treino
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss),
                        tf.assign_add(global_step, 1))
    #retorna o sub grafo do estimator
    return tf.estimator.EstimatorSpec(
                            mode=mode,
                            predictions=y,
                            loss=loss,
                            train_op=train)

estimator = tf.estimator.Estimator(model_fn=model_fn)
#define datasets
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])

x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

#criando os datasets, informa numero de epocas e tamanho do dataset
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

#treino
estimator.train(input_fn=input_fn, steps=1000)

#avalia modelo
#testa com os dados de treino
train_metrics = estimator.evaluate(input_fn=train_input_fn)
#testa com os dados de teste
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)