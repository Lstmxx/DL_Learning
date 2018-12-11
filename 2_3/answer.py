import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import *
import cv2
np.random.seed(1)

# y_hat = tf.constant(36,name='y_hat')
# y = tf.constant(39,name='y')

# loss = tf.Variable((y-y_hat)**2,name='loss') 

# init = tf.global_variables_initializer()

# with tf.Session() as session:
#     session.run(init)
#     print(session.run(loss))
def linear_function():
    np.random.seed(1)

    x = tf.constant(np.random.randn(3,1),name = "x")
    w = tf.constant(np.random.randn(4,3),name = "w")
    b = tf.constant(np.random.randn(4,1),name = "b")

    Y = tf.add(tf.matmul(w,x),b)

    with tf.Session() as session:
        cost = session.run(Y)

    return cost
def sigmoid(z):
    x = tf.placeholder(tf.float32,name="x")
    sigmoid = tf.sigmoid(x)
    with tf.Session() as session:
        result = session.run(sigmoid,feed_dict={x:z})
    return result
#print( "result = " + str(linear_function()))

def cost(logits,labels):
    z = tf.placeholder(tf.float32,name="logits")
    y = tf.placeholder(tf.float32,name="labels")

    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z,labels=y)
    with tf.Session() as session:
        cost = session.run(cost,feed_dict={z:logits,y:labels})
    return cost

def create_placeholders(n_x,n_y):
    X = tf.placeholder(tf.float32,shape=(n_x,None),name="X")
    Y = tf.placeholder(tf.float32,shape=(n_y,None),name="Y")
    return X,Y

def initialize_parameters(layers_dims):
    l = len(layers_dims)
    parameters = {}
    for i in range(1,l):
        parameters['W'+str(i)] = tf.get_variable('W'+str(i),[layers_dims[i],layers_dims[i-1]],initializer=tf.contrib.layers.xavier_initializer(seed = 1))
        parameters['b'+str(i)] = tf.get_variable('b'+str(i),[layers_dims[i],1],initializer=tf.zeros_initializer())
    return parameters

def forward_propagation(X,parameters):
    caches = []
    l = len(parameters)//2
    A=X
    for i in range(1,l):
        A_prew = A
        A = tf.nn.relu(tf.matmul(parameters['W'+str(i)],A_prew)+parameters['b'+str(i)])
    z3 = tf.matmul(parameters['W'+str(l)],A)+parameters['b'+str(l)]
    return z3
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

# X,Y = create_placeholders(X_train.shape[0],Y_train.shape[0])
# print ("X = " + str(X))
# print ("Y = " + str(Y))

tf.reset_default_graph()
layers_dims=[12288,25,12,6]
# with tf.Session() as seesion:
#     parameters = initalize_parameters(layers_dims)
#     print("W1 = " + str(parameters["W1"]))
#     print("b1 = " + str(parameters["b1"]))
#     print("W2 = " + str(parameters["W2"]))
#     print("b2 = " + str(parameters["b2"]))
#     print("W3 = " + str(parameters["W3"]))
#     print("b3 = " + str(parameters["b3"]))
with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters(layers_dims)
    Z3 = forward_propagation(X, parameters)
    print("Z3 = " + str(Z3))