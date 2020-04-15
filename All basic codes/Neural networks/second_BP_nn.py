import numpy as np
import math
import random
from matplotlib import pyplot as plt
### user defined for logic gates binary inputs
## XOR gate
Data_matrix=np.matrix('0 0 1 1;0 1 0 1')
bias=1
output=np.matrix('0 1 1 0')
count=100
## here the input nodes are fixed
N=input('Enter the desired no of nodes you need in hidden layer_For now enter two');
## we need an extra bias weight so it must be N+1,N+1
weights = np.random.randn(N,3)
out_weight=np.random.randn(N+1)
# define sigmoid function
def sigmoid(x):
  return 1 / (1 + math.exp(-x))
## training starts for the number of iterations given
for i in range(0,count):
    ## for the first layer input nodes are fixed as two
    ## the hidden nodes are user defined
    for j in range(0,4):
        inter=[]
        for k in range(N):
            Hidden = bias*weights[k,0]+ Data_matrix[0,j]*weights[k,1]+ Data_matrix[0,j]*weights[k,2]
            inter2 = sigmoid(Hidden)
            inter=np.append(inter,inter2)
        out_put= bias*out_weight[0]+inter[0]*out_weight[0]+inter[1]*out_weight[1]
        yout = sigmoid(out_put)
        delta3_1 = yout*(1-yout)*(output[0,j]-yout)
      # back propogation of an Neural Network
        delta_matrix=[]
        for k2 in range(N):
          delta2_1 = inter[k2]*(1-inter[k2])*out_weight[k2]*delta3_1
          delta_matrix=np.append(delta_matrix,delta2_1)
         
      ##updating starts here for all the nodes inlcuding output node
        weights[0,0] = weights[0,0] + (bias*delta_matrix[0])
        weights[1,0] = weights[1,0] + (bias*delta_matrix[1])
        out_weight[0] = out_weight[0] + (bias*delta3_1)
        weights[0,1] = weights[0,1] + (Data_matrix[0,j]*delta_matrix[0])
        weights[0,2] = weights[0,2] + (Data_matrix[0,j]*delta_matrix[0])
        weights[1,1] = weights[1,1] + (Data_matrix[1,j]*delta_matrix[1])
        weights[1,2] = weights[1,2] + (Data_matrix[1,j]*delta_matrix[1])
        out_weight[1] = out_weight[1] + (inter[0]*delta3_1)
        out_weight[2] =  out_weight[2]+ (inter[1]*delta3_1)


## testing starts here
test=np.matrix('0 0 1 1 ;0 1 0 1')
len=test.shape
yout=np.zeros((len[1]))
for j in range(0,len[1]):
    inter=[]
    for k in range(N):
        Hidden = bias*weights[k,0]+ test[0,j]*weights[k,1]+test[0,j]*weights[k,2]
        inter2 = sigmoid(Hidden)
        inter=np.append(inter,inter2)        
    out_put= bias*out_weight[0]+inter[0]*out_weight[0]+inter[1]*out_weight[1]
    yout[j] = sigmoid(out_put)

print (test)
print yout/sum(yout)

