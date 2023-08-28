# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 10:39:50 2020

@author: raneen_pc

This class is responsible for reading a time series data and preprocessing it 
and return a dictionary that contains the original data split into training 
and testing sets
"""


import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d


class ReadData:
    def __init__(self):  
        self = self
    
    def z_norm(self,x_train, x_test):
        # znorm
        std_ = x_train.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_
    
        std_ = x_test.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_
        
        return x_train, x_test

    def num_classes(self,y_train,y_test):
        #determine number of classes
        return len(np.unique(np.concatenate((y_train, y_test), axis=0)))
    
    def data_preparation(self,dataset_name, out_path):
        x_train, x_test,y_train,y_test = self.load_dataset_mat_form(dataset_name)
        max_length = self.get_func_length(x_train, x_test, func=max)
        n_var = x_train[0].shape[0]
        x_train = self.transform_to_same_length(x_train, n_var, max_length)
        x_test = self.transform_to_same_length(x_test, n_var, max_length)
        self.save_data_as_npy(out_path,x_train, y_train, x_test, y_test)
