import numpy as np 
import pandas as pd
import random


class model:
    def __init__(self,vocab_size : int, vector_size: int,encoded_data :int):
        self.w1 = np.random.rand(vocab_size,vector_size)
        self.w2 = np.random.rand(vector_size,vocab_size)
        
        
    def softmax(self,x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    
    def forward(self,encoded_data):
        if encoded_data.shape[0] != self.w1.shape[0]:
            raise ValueError("The input shape is not correct")
        self.z1 = np.dot(encoded_data,self.w1)
        self.z2 = np.dot(self.z1,self.w2)
        self.z3 = self.softmax(self.z2)
        return self.z3
    
    def backward(self,encoded_data,outputs,lr):
        self.dz2 = outputs - encoded_data
        self.dw2 = np.dot(self.z1.T,self.dz2)
        self.dz1 = np.dot(self.dz2,self.w2.T)
        self.dw1 = np.dot(encoded_data.T,self.dz1)
        self.w1 -= lr * self.dw1
        self.w2 -= lr * self.dw2 
        
    def train_for_a_epoch(self,encoded_data,lr):
        outputs = self.forward(encoded_data)
        self.backward(encoded_data,outputs,lr)
     


def encode_data(data,vocab_size):
    encoded_data = np.zeros((vocab_size,1))
    encoded_data[data] = 1
    return encoded_data


data = 