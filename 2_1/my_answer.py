import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from init_utils import sigmoid,relu,compute_loss,forward_propagation,backward_propagation
from init_utils import update_parameters,predict,load_dataset,plot_decision_boundary,predict_dec

# plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'

train_X,train_Y,test_X,test_Y = load_dataset()
def initialize_parameters_he(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)
    for l in range(1,L):
        parameters['W'+str(l)]=np.random.randn(layers_dims[l],layers_dims[l-1])*np.sqrt(2./layers_dims[l-1])
        parameters['b'+str(l)]=np.zeros((layers_dims[l],1))
    return parameters

def initialize_parameters_random(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)
    for l in range(1,L):
        parameters['W'+str(l)]=np.random.randn(layers_dims[l],layers_dims[l-1])
        parameters['b'+str(l)]=np.zeros((layers_dims[l],1))
    return parameters

def initialize_parameters_zeros(layers_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layers_dims)
    for l in range(1,L):
        parameters['W'+str(l)]=np.zeros((layers_dims[l],layers_dims[l-1]))
        parameters['b'+str(l)]=np.zeros((layers_dims[l],1))
    return parameters

def model(X,Y,learning_rate=0.01,num_iterations=15000,print_cost=True,initialization="zeros"):
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims=[X.shape[0],10,5,1]
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)
    for i in range(0,num_iterations):
        a3,cache=forward_propagation(X,parameters)

        cost = compute_loss(a3,Y)

        grads = backward_propagation(X,Y,cache)

        paramters=update_parameters(parameters,grads,learning_rate)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters
if __name__=="__main__":
    print("train:")
    p = model(train_X,train_Y,initialization="zeros")
    print("preditc on trein")
    pred = predict(train_X,train_Y,p)
    print ("predictions_train = " + str(pred))
    print("preditc on test")
    pred = predict(test_X,test_Y,p)
    print ("predictions_test = " + str(pred))
    plt.title("Model with large random initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5,1.5])
    axes.set_ylim([-1.5,1.5])
    plot_decision_boundary(lambda x: predict_dec(p, x.T), train_X, np.squeeze(train_Y))
