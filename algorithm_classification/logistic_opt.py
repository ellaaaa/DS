import pandas as pd
import numpy as np
import random
df = pd.read_table(r'C:\Users\Administrator\Desktop\HM\machinelearninginaction-master\Ch05\testSet.txt', header=None)
df.columns = ['a', 'b', 'C']

def sigmoid(data):
    return 1.0 / (1 + np.exp(-data))

def stoc_grad_ascent(data, labels, num_iter = 150):
    data_mat = np.mat(data)
    m, n = np.shape(data_mat)
    weights = np.ones(n)
    for j in range(num_iter):
        data_ind = range(m)
        for i in range(m):
            alpha = 4/(1.0+i+j)+0.01
            rand_ind = int(random.uniform(0,len(data_ind)))
            if type(weights) is np.matrixlib.defmatrix.matrix:
                h = sigmoid(weights[0]*data_mat[rand_ind].transpose())
            else:
                h = sigmoid(weights*data_mat[rand_ind].transpose())
            error = labels[rand_ind] - h
            weights = weights + alpha*error*data_mat[rand_ind]
    return weights

print(stoc_grad_ascent(df[['a','b']], df['C']))