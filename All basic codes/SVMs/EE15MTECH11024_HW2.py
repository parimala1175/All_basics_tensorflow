# Generate data for SVM classifier with L1 regularization
## using the solver from CVXPY
## solving it in terms of betas
## Author : EE15MTECH11024
## Parimala
import csv
import random
import math
import numpy as np
import math
import random
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
import scipy.stats
from cvxpy import *
## reading the data and formulating the data inorder to make it suitable for solvers
## Loading the Data matrix and reading
reader=csv.reader(open("t.csv","rb"),delimiter=',')
x=list(reader)
result_labels=np.array(x).astype('float')
print result_labels.shape
reader=csv.reader(open("X.csv","rb"),delimiter=',')
x=list(reader)
data_matrix=np.array(x).astype('float')
print data_matrix.shape
## Data framing
A_mat=[]
for k in xrange(len(result_labels)):
    temp = list(data_matrix[k])
    ##label inserting to find beta0 also with betas
    if result_labels[k][0] == 1:
        temp.append(1)
        A_mat.append(temp)			
    else:
        temp.append(-1)
        A_mat.append(temp)
A_mat = np.matrix(A_mat)
## standard way of solving the cvx problem in CVXPY
M,N = A_mat.shape
y = np.matrix([[1]]*M)
beta = Variable(N)
result_labels = np.array(result_labels)
Epsilon = Variable(M)
gamma=0.1 ## regularization Parameter
objective  = Minimize(sum_entries(square(beta))+gamma*sum_entries(square(Epsilon)))
## this is the formulation interms of betas with regularized beta objective function
constraints = [y <= A_mat*beta+np.diag(result_labels)*Epsilon]		
prob = Problem(objective,constraints)
prob.solve()
if prob.status == OPTIMAL:
    print 'training is done'
else:
    print 'The training set is not linearly separatable'
beta_out=np.matrix(beta.value)
print "Parameters"
print  np.matrix(beta.value)
### this completes the SVM optimization
## this gives out two beta values and one beta0 value
## for testing we  have to estimated betas and predict the output
## test data_set formulation
x_test=[[2,0.5,1],[-0.8,-0.7,1],[1.58,-1.33,1],[-0.008,0.001,1]]
#x_test = np.matrix(x_test)
Y_label = []
## Prediction starts From this
for x in x_test:
    a = np.array(x)*beta_out
    #print a
    if a > 0:
        ## if it is above the hyperplane 1
        Y_label.append([1])
    else:
        ## if it is below the hyperplane -1
        Y_label.append([-1])
print "For the given Test data Prdicted labels of SVM"
print Y_label

