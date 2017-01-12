import numpy as np
from pyDOE import *
import matplotlib.pyplot as plt
from sklearn import preprocessing


def sigmoid(z):
    return 1/(1.0 + np.exp(-z))

def L(x,y):
    return -(x - y)**2
    
def forecast(w, x):
    return sigmoid(w[0] * np.ones(10) + x * w[1:11]).mean()
    
    
    
"""
a = 2 * lhs(2, 400, criterion='maximin') - 1

plt.plot(a[:,0], a[:,1],'o')
plt.ylabel('some numbers')
plt.show()
"""

#Generating the train set and scaling it
#For function  (1 + sin(x))/2 

N = 100#number of points

Train_x = np.linspace(-np.pi, np.pi, N)
Train_y = ( 1.0 - np.sin(Train_x)) / 2.0

scaler = preprocessing.StandardScaler().fit(Train_x)
Train_x_scaled = scaler.transform(Train_x)

M = 10#number of pairs for each x
K = 10#number of instances W for each pair

Pairs = {}
Losses = {}

sigma = np.random.rand()

for i in range(len(Train_x_scaled)):
    sigma = np.random.rand()
    print( str(i) + "step")
    v_set = 2.0 * lhs(11, M, criterion='maximin') - 1.0
    error_to_pair = {}
    #for each pair we generate matrices
    
    for j in range(M):#For every pair of parameters
        SetW = [0 for q in range(K)]
        
        for k in range(K):#Generate matrix
            Diag = [[sigma if r==s else 0 for r in range(11)] for s in range(11)]
            SetW[k] = np.random.multivariate_normal((v_set[j,:]),Diag,1)
            
        
        losses = [0 for q in range(K)]
        
        for q in range(K):#loss on matrix k
            losses[q] =  L(sigmoid(SetW[q][0,0] * np.ones(10) + Train_x_scaled[i] * SetW[q][0,1:11]).mean(), Train_y[i])    
            
        Average_error = (np.array(losses)).mean()
        error_to_pair[Average_error] = v_set[j,:]
    
    min_error = max(error_to_pair.keys())
    Pairs[Train_x_scaled[i]] = error_to_pair[min_error]
    Losses[Train_x_scaled[i]] = min_error   
        



s = np.zeros(11)

for i in Pairs.values():
    s = s + i / len(Pairs.keys())
  

predictions = np.zeros(N)

for i in range(N):
    predictions[i] = forecast(s, Train_x_scaled[i])
    

print("!!!")  
plt.plot(Train_x_scaled, predictions,'o', Train_x_scaled, Train_y )
plt.ylabel('some numbers')
plt.show()

