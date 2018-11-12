import numpy as np 
from sklearn import datasets,linear_model
import sklearn
from matplotlib import pyplot as plt
def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.get_cmap("Spectral"))
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=np.squeeze(y), cmap=plt.cm.get_cmap("Spectral"))
    plt.show()

def loat_planar_dataset():
    np.random.seed(1)
    m = 400
    N=int(m/2)
    D=2
    X=np.zeros((m,D))
    Y=np.zeros((m,1),dtype='uint8')
    a=4
    for j in range(2):
        ix=range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N)+np.random.randn(N)*0.2
        r=a*np.sin(4*t)+np.random.randn(N)*0.2
        X[ix]=np.c_[r*np.sin(t),r*np.cos(t)]
        Y[ix]=j
    X = X.T
    Y = Y.T
    return X,Y
def normalizeRows(x):
    x_norm = np.linalg.norm(x, axis=1, keepdims = True)
    x = x / x_norm
    return x
def sigmoid(x):
    x=1.0/(1+1/np.exp(x))
    return x
def d_sigmoid(s_x):
    d_x = s_x(1-s_x)
    return d_x
# X,Y=loat_planar_dataset()
# plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.get_cmap("Spectral"))
# plt.show()
# shape_X=np.shape(X)
# shape_Y=np.shape(Y)
# m=np.shape(X[0,:])
# print ('The shape of X is: ' + str(shape_X))
# print ('The shape of Y is: ' + str(shape_Y))
# print ('I have m = %d training examples!' % (m))

# clf = linear_model.LogisticRegressionCV()
# clf.fit(X.T,Y.T.ravel())

# plot_decision_boundary(lambda x: clf.predict(x), X, Y)
# plt.title("Logistic Regression")
# LR_predictions = clf.predict(X.T)
# print ('Accuracy of logistic regression: %d ' % 
# float((np.dot(Y,LR_predictions) + 
# np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
#        '% ' + 
#        "(percentage of correctly labelled datapoints)")#Y*Y_hat+(1-Y)*(1-Y_hat)#**看预测和真实匹配程度**
