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
    l = len(parameters)//2
    A=X
    for i in range(1,l):
        A_prew = A
        A = tf.nn.relu(tf.matmul(parameters['W'+str(i)],A_prew)+parameters['b'+str(i)])
    z3 = tf.matmul(parameters['W'+str(l)],A)+parameters['b'+str(l)]
    return z3

def compute_cost(Z3, Y):
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = labels))
    ### END CODE HERE ###
    
    return cost

def model(X_train,Y_train,X_test,Y_test,layers_dims=[12288,25,12,6],learning_rate=0.0001,num_it=1500,minibatch_size = 32,print_cost = True):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x,m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    X,Y = create_placeholders(n_x,n_y)
    parameters = initialize_parameters(layers_dims)
    z3 = forward_propagation(X,parameters)
    cost = compute_cost(z3,Y)
    OPT = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        for i in range(num_it):
            e_cost = 0
            num_minibatches = int(m/minibatch_size)
            seed = seed + 1 
            minibatches = random_mini_batches(X_train,Y_train,mini_batch_size=minibatch_size,seed=seed)
            for minibatche in minibatches:
                (minibatch_x,minibatch_y) = minibatche
                _,minibatch_cost = session.run([OPT,cost],feed_dict={X:minibatch_x,Y:minibatch_y})
                e_cost += minibatch_cost/num_minibatches
            if print_cost == True and i % 100 ==0:
                print("Cost after epoch %i: %f" % (i,e_cost))
            if print_cost == True and i % 5 == 0:
                costs.append(e_cost)
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate = "+str(learning_rate))
        plt.show()

        parameters = session.run(parameters)
        print ("Parameters have been trained!")
        correct_prediction = tf.equal(tf.argmax(z3), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        return parameters


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

# layers_dims=[12288,25,12,6]
parameters = model(X_train, Y_train, X_test, Y_test)