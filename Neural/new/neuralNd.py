import scipy 
import numpy as np
from pyDOE import *
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import tree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sklearn.metrics
import time




input_size = 8

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def Loss(x,y):
    return (x - y)**2




    
#sigmoid!!
def forecast(w, x, c, input_size):

    return activation(w[5 * (2 + input_size)] + (activation( w[5 * input_size: 5 * (input_size + 1)] + np.matrix(x) * w[:5 * input_size].reshape(input_size, 5), c)).dot(w[5 * (1 + input_size):5 * (2 + input_size)]), c)    
    
def act1(z):
    return (1.0 + np.tanh(z))/2.0
    
def act2(z):
    return ((2.0 * np.arctan(z)/np.pi) + 1.0)/2.0
    
def activation(z, c):    
    if (c == 'arctan'):
        return act2(z)
    if (c == 'tanh'):
        return act1(z)
    return sigmoid(z)

#Generating the train set and scaling it
#For function  (1 + sin(x))/2 

#N = number of points
#M = number of generated expectations for each x
#K = number of instances W for each expectation

def learn_network(c, M = 100, K = 20):
    
    #we generate the training set
    #Train_x = np.linspace(-np.pi, np.pi, N)
    #Train_x_scaled = np.linspace(-1, 1, N)
    #Train_y = ( 1.0 + np.sin(Train_x_scaled)) / 2.0
    
    
    
    """
    #X
    f1 = open('madelon_train.data.txt', 'r')
    c = f1.readlines()
    #!!!!!!!!!!len(c)
    L = [c[i].split(" ")[:input_size] for i in range(1000)]
    L = [[float(L[i][j]) for j in range(len(L[i]))] for i in range(len(L))]
    Train_x = np.matrix(L)
    f1.close()
    
    #Y
    #function to approximate
    f2 = open('madelon_train.labels.txt', 'r')   
    
    c = f2.readlines()
    #len(c)
    L = [float(c[i].split("\n")[0]) for i in range(1000)]
    L = [[1.0] if L[i] == 1.0 else [0.0] for i in range(len(L))]
    Train_y = np.matrix(L)
    f2.close()
    
    """
    f = open("pima-indians-diabetes.data.txt",'r')
    l = f.readlines()
    l = [l[i].split("\n") for i in range(300)]
    l = [l[i][0].split(',') for i in range(len(l))]
    l = [[float(l[i][j]) for j in range(len(l[i]))] for i in range(len(l))]
    
    Train_x = np.matrix([l[i][:8] for i in range(len(l))])
    Train_y = np.matrix([[l[i][8]] for i in range(len(l))])
    #Train_y = ( np.multiply(Train_x,Train_x) * np.matrix([[1],[1]])) / 2.0

    #we scale dataset
    scaler = preprocessing.StandardScaler().fit(Train_x);
    Train_x_scaled = scaler.transform(Train_x);
    
    
    #Dictionary of type {point:best_expectation_of_parameters}
    Pairs = {}
    
    #Disctionary of type {}
    Losses = {}
    
    #Variance parameter
    sigma = 1.0
    Diag = [[sigma if r==s else 0 for r in range(5 * (2 + input_size) + 1)] for s in range(5 * (2 + input_size) + 1)]
    #for each x we find best fitting vector of expectation
    
    
    
    for i in range(len(Train_x_scaled)):
        
        
        print( str(i) + "step")
        
        #we generate a random set of parameters and scale them to [0,1]^10
        
        v_set = 2.0 * lhs(5 * (2 + input_size) + 1, M, criterion='maximin') - 1.0
        #maps average error to expectation_of_parameters(vector)
        error_to_pair = {}
         
        for j in range(M):#For every vector of parameters(expectation)
        
            #SECTION 1
            
            #SetW = [0 for q in range(K)]
            
            SetW = np.random.multivariate_normal((v_set[j,:]), Diag, K)
                       
            #losses = [0 for q in range(K)]
            
            
            #for q in range(K):#we calculate loss on each instance of W
                #sigmoid!!!   
            losses =  map(lambda A: Loss(activation(A[5 * (2 + input_size)] +\
            activation(A[5 * (input_size) : 5 * (1 + input_size)] +\
            np.matrix(Train_x_scaled[i]) * A[ : 5 *  input_size].reshape(input_size,5), c).dot(A[5 * (1\
            + input_size):5*(2 + input_size)]), c), Train_y[i,0]), SetW )
                
                
            #we calculate average error for this instance of expectation vector
            Average_error = (np.array(losses)).mean()
            
            #error -> parameter of exspectation realising it
            error_to_pair[Average_error] = v_set[j,:]
            
            
        
        #we find the expectation that gives the minimum expected error and attach it the x 
        min_error = min(error_to_pair.keys())
        
        Pairs[i] = error_to_pair[min_error]
        Losses[i] = min_error   
        
    print("!!!!!!!!!!!")
    clf = tree.DecisionTreeRegressor(max_depth = 8);

    clf = clf.fit([[Train_x_scaled[i][j] for j in range(input_size)] for i in range(Train_x_scaled.shape[0])], [Pairs[i] for i in range(Train_x_scaled.shape[0])]);
    
    test(scaler, Train_x_scaled, Train_y, clf, c, input_size)    
    #return (scaler, Train_x_scaled, Train_y, Pairs, clf)
        

def test(scaler, Train_x_scaled, Train_y, clf, c, input_size):
    
    #Test_x_1 = np.linspace(-1.0,1.0 ,13)
    #Test_x_2 = np.linspace(-1.0, 1.0, 13)
    #Test_x = np.matrix([[x,y] for x in Test_x_1 for y in Test_x_2])
    #Test_y = ( 1.0 + np.multiply(Test_x,Test_x) * np.matrix([[1],[1]])) / 2.0

    #Test_x_scaled = scaler.transform(Test_x);
    #Test_x = np.linspace(-np.pi, np.pi, 23)
    
    #Test_x_scaled = scaler.transform(Test_x)
    
    """
    f1 = open('madelon_valid.data.txt', 'r')
    c = f1.readlines()
    L = [c[i].split(" ")[:input_size] for i in range(len(c))]
    L = [[float(L[i][j]) for j in range(len(L[i]))] for i in range(len(L))]
    Test_x = np.matrix(L)
    Test_x_scaled = scaler.transform(Test_x);
    f1.close()
    
    #Y
    #function to approximate
    f2 = open('madelon_valid.labels.txt', 'r')   
    
    c = f2.readlines()
    L = [float(c[i].split("\n")[0]) for i in range(len(c))]
    L = [1.0 if L[i] == 1.0 else 0.0 for i in range(len(L))]
    Test_y = L
    f2.close()
    """  
    
    f = open("pima-indians-diabetes.data.txt",'r')
    l = f.readlines()
    l = [l[i].split("\n") for i in range(300,len(l))]
    l = [l[i][0].split(',') for i in range(len(l))]
    l = [[float(l[i][j]) for j in range(len(l[i]))] for i in range(len(l))]    
    
    
    Test_x = np.matrix([l[i][:8] for i in range(len(l))])
    Test_x_scaled = scaler.transform(Test_x);
    Test_y = np.matrix([[l[i][8]] for i in range(len(l))])
    
    pd = clf.predict([[x[i] for i in range(x.shape[0])] for x in Test_x_scaled])
    
    
    predictions = np.zeros(Test_x.shape[0])
    
    for i in range(Test_x.shape[0]):
        predictions[i] = forecast(pd[i], Test_x_scaled[i], c, input_size)
    

    print("LogLoss per unit is " + str(sklearn.metrics.log_loss(Test_y, predictions, normalize = True)));
    
    predictions = [1  if predictions[i] > 0.5 else 0 for i in range(len(predictions))]
    
    print("Precision is " + str(sklearn.metrics.accuracy_score(Test_y, predictions, normalize = True)));
    
    """
    print("Train_X********************")
    for i in range(Train_x_scaled.shape[0]):
        print(Train_x_scaled[i][0])
        
    print("Train_Y********************")
    
    for i in range(Train_x_scaled.shape[0]):
        print(Train_x_scaled[i][1])
     
    print("Train_Z********************")
    for i in range(Train_x_scaled.shape[0]):
        print(Train_y[i][0,0])
        
    print("Test_X********************")
    for i in range(Test_x_scaled.shape[0]):
        print(Test_x_scaled[i][0])
        
    print("Test_Y********************")
    
    for i in range(Test_x_scaled.shape[0]):
        print(Test_x_scaled[i][1])
    """
    #print("Predictions " + c + "!********************")
    #for i in range(Test_x_scaled.shape[0]):
    #    print(predictions[i])
    
    

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    
    #ax.plot_trisurf([Train_x_scaled[i][0] for i in range(Train_x_scaled.shape[0])],[Train_x_scaled[i][1] for i in range(Train_x_scaled.shape[0])],[x[0,0] for x in Train_y],cmap=cm.bone, alpha=0.4 )
    #ax.scatter([Test_x_scaled[i][0] for i in range(Test_x_scaled.shape[0])],[Test_x_scaled[i][1] for i in range(Test_x_scaled.shape[0])],predictions, s = 3, c = "g")
    #fig.savefig(c + "3d.png")
    #plt.plot(Test_x_scaled, predictions,'o', Train_x_scaled, Train_y)
    #plt.title("Activation function: " + c)
    #plt.savefig(c + "3d.png")

def read_data():
    
    f = open('madelon_train.data.txt', 'r')
    
    c = f.readlines()
    L = [c[i].split(" ")[:500] for i in range(len(c))]
    L = [[float(L[i][j]) for j in range(len(L[i]))] for i in range(len(L))]
    
    
    #for y
    L = [float(c[i].split("\n")[0]) for i in range(len(c))]

#for i in Pairs.values():
#    s = s + i / len(Pairs.keys())
  
#np.array(Pairs.values()).mean(1)

#forecast(Pairs[Train_x_scaled[i]]
"""
s = np.zeros(10)
for i in range(N):
    predictions[i] = forecast(pd[i], Train_x_scaled[i])
    

print("!!!")  
plt.plot(Train_x_scaled, predictions,'o', Train_x_scaled, Train_y )
plt.ylabel('some numbers')
plt.show()

"""

def network():
    
    
    from sklearn.neural_network import MLPClassifier

    
    
    f = open("pima-indians-diabetes.data.txt",'r')
    l = f.readlines()
    l = [l[i].split("\n") for i in range(300)]
    l = [l[i][0].split(',') for i in range(len(l))]
    l = [[float(l[i][j]) for j in range(len(l[i]))] for i in range(len(l))]
    f.close()
    
    Train_x = np.matrix([l[i][:8] for i in range(len(l))])
    Train_y = np.matrix([[l[i][8]] for i in range(len(l))])
    #Train_y = ( np.multiply(Train_x,Train_x) * np.matrix([[1],[1]])) / 2.0

    #we scale dataset
    scaler = preprocessing.StandardScaler().fit(Train_x)
    Train_x_scaled = scaler.transform(Train_x)
    
    
    clf = MLPClassifier(activation='tanh', solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5))
    
    clf.fit(Train_x_scaled, Train_y)
    
    f = open("pima-indians-diabetes.data.txt",'r')
    l = f.readlines()
    l = [l[i].split("\n") for i in range(300, len(l))]
    l = [l[i][0].split(',') for i in range(len(l))]
    l = [[float(l[i][j]) for j in range(len(l[i]))] for i in range(len(l))]
    
    Test_x = np.matrix([l[i][:8] for i in range(len(l))])
    Test_y = np.matrix([[l[i][8]] for i in range(len(l))])
    Test_x_scaled = scaler.transform(Test_x)
    f.close()    
    
    prediction = clf.predict(Test_x_scaled)
    
    print("Precision is " + str(sklearn.metrics.accuracy_score(Test_y, prediction, normalize = True)))
    
    
    
    
    
    
    
    
    



