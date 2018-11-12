import numpy as np 
from sklearn import datasets,linear_model
import sklearn
from matplotlib import pyplot as plt
from planar_utils import * 

def layer_sizes(X,Y):
    n_x=X.shape[0]
    n_y=Y.shape[0]
    n_h=4
    return (n_x,n_h,n_y)
def initalize_paramters(n_x,n_h,n_y):
    np.random.seed(2)
    W1=np.random.randn(n_h,n_x)*0.01
    b1=np.zeros((n_h,1))
    W2=np.random.randn(n_y,n_h)*0.01
    b2=np.zeros((n_y,1))
    assert(W1.shape==(n_h,n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    parameters={'W1':W1,
                'b1':b1,
                'W2':W2,
                'b2':b2}
    return parameters
def forward_progagation(X,parameters):
    W1=parameters['W1']
    b1=parameters['b1']
    W2=parameters['W2']
    b2=parameters['b2']

    Z1 = np.dot(W1,X)+b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1)+b2
    A2 = sigmoid(Z2)

    assert(A2.shape==(1,X.shape[1]))

    cache={
        'Z1':Z1,
        'A1':A1,
        'Z2':Z2,
        'A2':A2
    }

    return A2,cache
def compute_cost(A2,Y,parameters):
    m = Y.shape[1]
    cost = -1/m*np.sum(
    np.multiply(Y,np.log(A2))+
    np.multiply(1-Y,np.log(1-A2))
    )
    assert(isinstance(cost,float))
    return cost

def backward_progagation(parameters,cache,X,Y):
    m=X.shape[1]
    W1=parameters['W1']
    b1=parameters['b1']
    W2=parameters['W2']
    b2=parameters['b2']
    Z1=cache['Z1']
    A1=cache['A1']
    Z2=cache['Z2']
    A2=cache['A2']

    dZ2 = A2 - Y
    dw2=np.dot(dZ2,A1.T)/m
    db2=np.sum(dZ2,axis=1,keepdims=True)/m
    dz1=np.multiply(np.dot(W2.T,dZ2),(1-A1**2))
    dw1=np.dot(dz1,X.T)
    db1=np.sum(dz1,axis=1,keepdims=True)/m

    grads={
        "dW1":dw1,
        "db1":db1,
        "dW2":dw2,
        "db2":db2
    }
    return grads

def update_parameters(parameters,grads,learning_rate=1.2):
    W1=parameters["W1"]
    W2=parameters["W2"]
    b1=parameters["b1"]
    b2=parameters["b2"]
    dW1=grads["dW1"]
    dW2=grads["dW2"]
    db1=grads["db1"]
    db2=grads["db2"]

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def nn_model(X,Y,n_h,num_iterations=10000,print_cost=False):
    np.random.seed(3)
    n_x,n_hh,n_y=layer_sizes(X,Y)

    parameters = initalize_paramters(n_x,n_h,n_y)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    for i in range(0,num_iterations):
        A2,cache=forward_progagation(X,parameters)#前向传播节点
        cost = compute_cost(A2, Y, parameters)#计算损失函数
        grads=backward_progagation(parameters,cache,X,Y)#计算后向传播梯度
        parameters=update_parameters(parameters,grads,learning_rate=1.2)#使用梯度更新W，b一次

        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

def predict(parameters,X):
    A2, cache = forward_progagation(X, parameters)
    predictions = (A2 > 0.5)
    return predictions

X,Y=loat_planar_dataset()
parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)
# Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()
predictions=predict(parameters,X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')