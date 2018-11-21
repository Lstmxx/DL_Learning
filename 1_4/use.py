#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import time 
import numpy as np
import h5py
import scipy
from matplotlib import pyplot as plt
from scipy import ndimage
from PIL import Image
from dnn_utils_v2 import *
#from dnn_app_utils_v2 import *
import cv2
import json
if __name__=="__main__":
    plt.rcParams['figure.figsize'] = (5.0,4.0)
    plt.rcParams['image.interpolation']='nearest'
    plt.rcParams['image.cmap']='gray'
    train_x_orig,train_y,test_x_orig,test_y,classes = load_data()
    # index = 7
    # print(train_y.shape)
    # print(type(classes))
    # print(classes.shape)
    # plt.imshow(train_x_orig[index])
    # #cv2.imshow("d",train_x_orig[index])
    # #cv2.waitKey(0)
    # print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")
    layers_dims = [12288, 20, 7, 5, 1]#5层
    train_x,test_x=normalization_inputdata(train_x_orig,test_x_orig)
    parameters = L_layer_model(train_x,train_y,layers_dims,print_cost=True)

    p = predict(test_x,test_y,parameters)