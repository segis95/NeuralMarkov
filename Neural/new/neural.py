import numpy as np
from pyDOE import *
import matplotlib.pyplot as plt
from sklearn import preprocessing


def sigmoid(z):
    return 1/(1.0 + np.exp(-z))

def L(x,y):
    return (x - y)**2
    
"""
a = 2 * lhs(2, 400, criterion='maximin') - 1

plt.plot(a[:,0], a[:,1],'o')
plt.ylabel('some numbers')
plt.show()
"""

#Generating the train set and scaling it
#For function  (1 + sin(x))/2 

N = 100

Train_x = np.linspace(-np.pi, np.pi, N)
Train_y = ( 1.0 + np.sin(Train_x)) / 2.0

scaler = preprocessing.StandardScaler().fit(Train_x)
Train_x_scaled = scaler.transform(Train_x)

M = 1000
K = 1000

Pairs = {}
Losses = {}

for i in range(len(Train_x_scaled)):
    sigma = np.random.rand()
    v_set = 2 * lhs(11, M, criterion='maximin') - 1
    #for each pair we generate matrices
    for j in range(M):
        SetW = [0 for q in range(K)]
        for k in range(K):
            Diag = [[sigma if i==j else 0 for i in range(11)] for j in range(11)]
            SetW[k] = np.random.multivariate_normal((v_set[j,:]),Diag,1)
            
        b = SetW[k][0,0] * np.ones(10)
        
        losses = [0 for q in range(K)]
        for k in range(K):
            
            losses[q] =  L(SetW[q][0,0] * np.ones(10) + Train_x_scaled[i] * SetW[q][0,1:11], Train_y[i]).sum()       
            
        q = np.argmin(losses)
        Pair[]
        
        
print()





