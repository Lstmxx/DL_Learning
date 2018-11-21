#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import numpy as np
from matplotlib import pyplot as plt
import h5py

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A,cache

def relu(Z):
    cache = Z
    A = np.maximum(0,Z)
    assert(A.shape==Z.shape)
    return A,cache

def relu_backward(dA,cache):
    dZ = np.array(dA,copy=True)
    Z = cache
    dZ[Z<=0] = 0
    assert(dZ.shape==Z.shape)
    return dZ

def sigmoid_backward(dA,cache):
    Z=cache
    s = 1/(1+np.exp(-Z))
    dZ = dA*s*(1-s)
    assert(dZ.shape==Z.shape)
    return dZ

def load_data():
    train_dataset = h5py.File('/home/negispringfield/Desktop/DL_Learning/1_4/datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
    
    test_dataset = h5py.File('/home/negispringfield/Desktop/DL_Learning/1_4/datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
    
    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))  
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(1)
    W1=np.random.randn(n_h,n_x)*0.01
    b1=np.random.randn(n_h,1)
    W2=np.random.randn(n_y,n_h)*0.01
    b2=np.random.randn(n_y,1)
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2} 
    return parameters 

def initialize_parameters_deep(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)
    for l in range(1,L):
        parameters['W'+str(l)]=np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
        #parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) 
        parameters['b'+str(l)]=np.zeros((layer_dims[l],1))
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))   
    return parameters

def linear_forward(A,W,b):
    cache = {
        'W':W,
        'b':b,
        'A':A
    }
    Z = W.dot(A)+b
    assert(Z.shape==(W.shape[0],A.shape[1]))
    return Z,cache

def linear_activation_forward(A_prew,W,b,activation):
    if activation == 'relu':
        Z,linear_cache = linear_forward(A_prew,W,b)
        A,linear_activation_cache = relu(Z)
    if activation == 'sigmoid':
        Z,linear_cache = linear_forward(A_prew,W,b)
        A,linear_activation_cache = sigmoid(Z)
    cache = {
        'linear_cache':linear_cache,
        'linear_activation_cache':linear_activation_cache
    }
    assert(A.shape==(W.shape[0],A_prew.shape[1]))
    return A,cache

def L_model_forward(X,parameters):
    caches = []
    A=X
    L=len(parameters)//2
    for l in range(1,L):
        A_prew = A
        A,cache = linear_activation_forward(A_prew,parameters['W'+str(l)],parameters['b'+str(l)],activation='relu')
        caches.append(cache)
    AL,cache = linear_activation_forward(A,parameters['W'+str(L)],parameters['b'+str(L)],activation='sigmoid')
    caches.append(cache)
    assert(AL.shape==(1,X.shape[1]))
    return AL,caches

def compute_cost(AL,Y):
    m=Y.shape[1]
    #cost = -(Y*np.log(AL)+(1-Y)*np.log(1-AL))/m
    #cost = -1./m*(np.dot(Y,np.log(AL.T))+np.dot(1-Y,np.log(1-AL).T))
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))

    cost = np.squeeze(cost)
    assert(cost.shape == ())
    return cost

def linear_backward(dZ, cache):
    A_prev = cache['A']
    W = cache['W']
    b = cache['b']
    m=A_prev.shape[1]
    dW=1.0/m*np.dot(dZ,A_prev.T)
    db=1.0/m*np.sum(dZ,axis=1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev,dW,db

def linear_activation_backward(dA,cache,activation):
    linear_cache=cache['linear_cache']
    activation_cache=cache['linear_activation_cache']
    if activation=='relu':
        dZ = relu_backward(dA,activation_cache)
        dA_prev,dW,db = linear_backward(dZ,linear_cache)
    if activation=='sigmoid':
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev,dW,db = linear_backward(dZ,linear_cache)
    return dA_prev, dW, db

def L_model_backward(AL,Y,caches):
    grads={}
    L=len(caches)
    Y=Y.reshape(AL.shape)
    dAL=-(np.divide(Y,AL)-np.divide(1-Y,1-AL))
    current_cache = caches[L-1]
    grads["dA"+str(L)],grads["dW"+str(L)],grads["db"+str(L)]=linear_activation_backward(dAL,current_cache,activation='sigmoid')
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp,dW_temp,db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

def update_parameters(parameters,grads,learning_rate):
    L = len(parameters)//2
    for l in range(L):
        parameters['W'+str(l+1)]=parameters['W'+str(l+1)] - learning_rate*grads['dW'+str(l+1)]
        parameters['b'+str(l+1)]=parameters['b'+str(l+1)] - learning_rate*grads['db'+str(l+1)]
    return parameters

def predict(X,y,parameters):
    m=X.shape[1]
    n=len(parameters)//2
    p=np.zeros((1,m))

    probas,caches = L_model_forward(X,parameters)
    for i in range(0,probas.shape[1]):
        if probas[0,i]>0.5:
            p[0,i]=1
        else:
            p[0,i]=0
    print("Accuracy: "  + str(np.sum((p == y)/float(m))))
    return p

def L_layer_model(X,Y,layers_dims,learning_rate=0.0075,num_iteration=3000,print_cost=False):
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters_deep(layers_dims)
    for i in range(0,num_iteration):
        AL,caches = L_model_forward(X,parameters)
        cost=compute_cost(AL,Y)
        grads = L_model_backward(AL,Y,caches)
        parameters = update_parameters(parameters,grads,learning_rate)
        if print_cost and i % 100 == 0: 
            print ("Cost after iteration %i: %f" %(i, cost)) 
        if print_cost and i % 100 == 0: 
            costs.append(cost) 
    plt.plot(np.squeeze(costs)) 
    plt.ylabel('cost') 
    plt.xlabel('iterations (per tens)') 
    plt.title("Learning rate =" + str(learning_rate)) 
    plt.show()

    return parameters

def normalization_inputdata(train_x_orig,test_x_orig):
    train_x_f = train_x_orig.reshape(train_x_orig.shape[0],-1).T
    test_x_f = test_x_orig.reshape(test_x_orig.shape[0],-1).T
    train_x = train_x_f/255
    test_x = test_x_f/255
    return train_x,test_x
