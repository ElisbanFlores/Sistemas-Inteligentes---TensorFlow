# Problema de clasificacion con Redes Neuronales
# Utilizando TensorFlow
# Car Evaluation: https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
# Autor Elisban Flores Quenaya
# MSC UCSP - Arequipa
# Enero del 2017

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# definiendo la funcion add_Layer

def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):    
    Weights = tf.Variable(tf.truncated_normal([in_size,out_size],stddev=0.1), name="W")    
    biases = tf.Variable(tf.zeros([1,out_size])+0.1, name="b")
    Wx_plus_b = tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)    
    return outputs

# calcular la precision

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction=tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs: v_xs, ys: v_ys})
    return result
    

# Lectura de datos
# lectura de archivos car.data y  car-prueba.data
data = pd.read_csv("car.data")
data_test = pd.read_csv("car-prueba.data")

features = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]

buying_val={"vhigh":0,"high":1,"med":2,"low":3}
maint_val={"vhigh":0,"high":1,"med":2,"low":3}
doors_val={"2":0,"3":1,"4":2,"5more":3}
persons_val={"2":0,"4":1,"more":2}
lug_boot_val={"small":0,"med":1,"big":2}
safety_val={"low":0,"med":1,"high":2}

class_val={"unacc":0,"acc":1,"good":2,"vgood":3}

def ind_buying(c): return buying_val[c]
def ind_maint(c): return maint_val[c]
def ind_doors(c): return doors_val[c]
def ind_persons(c): return persons_val[c]
def ind_lug_boot(c): return lug_boot_val[c]
def ind_safety(c): return safety_val[c]
def ind_class(c):  return class_val[c]

# configurando a valores numericos para la data de entrenamiento
data["_buying"]= data["buying"].apply(ind_buying)
data["_maint"]= data["maint"].apply(ind_maint)
data["_doors"]= data["doors"].apply(ind_doors)
data["_persons"]= data["persons"].apply(ind_persons)
data["_lug_boot"]= data["lug_boot"].apply(ind_lug_boot)
data["_safety"]= data["safety"].apply(ind_safety)
data["_class"]= data["class"].apply(ind_class)

# configurando a valores numericos para la data de test

data_test["_buying"]= data_test["buying"].apply(ind_buying)
data_test["_maint"]= data_test["maint"].apply(ind_maint)
data_test["_doors"]= data_test["doors"].apply(ind_doors)
data_test["_persons"]= data_test["persons"].apply(ind_persons)
data_test["_lug_boot"]= data_test["lug_boot"].apply(ind_lug_boot)
data_test["_safety"]= data_test["safety"].apply(ind_safety)
data_test["_class"]= data_test["class"].apply(ind_class)

# preparando data para las capas de la red

#data_test["tmp"]= data_test["class"].apply(help)
data_x=data.drop(["buying", "maint", "doors", "persons", "lug_boot","safety","class"], axis=1)
data_test_x=data_test.drop(["buying", "maint", "doors", "persons", "lug_boot","safety","class"], axis=1)


data_x.loc[:,("y1")]=data_x["_class"]==0
data_x.loc[:,("y1")]=data_x["y1"].astype(int)
data_x.loc[:,("y2")]=data_x["_class"]==1
data_x.loc[:,("y2")]=data_x["y2"].astype(int)
data_x.loc[:,("y3")]=data_x["_class"]==2
data_x.loc[:,("y3")]=data_x["y3"].astype(int)
data_x.loc[:,("y4")]=data_x["_class"]==3
data_x.loc[:,("y4")]=data_x["y4"].astype(int)
print (data_x)

# la data en el formato requerido, matrices
x_data = data_x.loc[:,["_buying", "_maint", "_doors", "_persons", "_lug_boot","_safety"]].as_matrix()
y_data = data_x.loc[:,["y1","y2","y3","y4"]].as_matrix()

x_datatest=data_test_x.loc[:,["_buying", "_maint", "_doors", "_persons", "_lug_boot","_safety"]].as_matrix()


#definiendo los placeholder para las entradas de la red neuronal

xs = tf.placeholder(tf.float32,[None,6]) # buying, maint, doors, persons, lug_boot, safety
ys = tf.placeholder(tf.float32,[None,4])

# add capas ocultas

layer1=add_layer(xs,6,10,n_layer=1, activation_function=tf.nn.relu)

# Add capa de salida
prediction=add_layer(layer1,10,4,n_layer=2, activation_function=tf.nn.softmax)

# Para el entrenamiento

# el error entre los datos predecidos y los datos reales
loss=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1])) 
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

sess=tf.Session()
sess.run(tf.global_variables_initializer())


for i in range (500):
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    if i%50==0:
        print(compute_accuracy(x_data,y_data))

# Para la prediccion

y_pre = sess.run(prediction, feed_dict={xs: x_datatest})

# imprimir resultados

print("Data de prueba")
print("Datos ingresados")
print(data_test)
print("datos para la prediccion" )
print(x_datatest) 
print("resultados")
print("        unacc            acc              good              vgood")
print(y_pre)


