import numpy as np
import math
import random
from matplotlib import pyplot as plt
from scipy import linalg
mean = [0, 0]
cov = [[1, 0.9], [0.9, 1]]  # diagonal covariance
x, y1 = np.random.multivariate_normal(mean, cov, 5000).T
plt.plot(x, y1, 'x')
plt.axis('equal')
plt.pause(1)
Y=np.zeros([2,5000])
for i in range(5000):
    Y[0][i]=x[i]
    Y[1][i]=y1[i]
### I have generated My own data of 2d with some mean and variance
##  then plotted adnd changed the dimension to the optimal variance dimension using PCA
def PCA(Y):
    [m,N]=Y.shape
    Corr_matrix=np.zeros([m,m],'float')
    YT=np.transpose(Y)
    for i in range(m):
        r=Y[i,:]
        for j in range(m):
            r2=YT[:,j]
            Corr_matrix[i][j]=np.dot(r,r2)/float(N)
    print Corr_matrix
    ## Eigen value Decomposition
    D,V=linalg.eig(Corr_matrix)
    Transform=np.transpose(V)
    W=np.dot(V,Y)
    plt.plot(W[0,:],W[1,:], 'x')
    
    plt.show()
    return W;
p=PCA(Y)
[m,N]=Y.shape
Corr_matrix2=np.zeros([m,m],'float')
YT2=np.transpose(p)
## just for validating PCA
## we found the covariance at the for modified Dimension
## observed that we have a diagonal Matrix
## first element in the diagonal Matrix is having more value compared to another that shows that
## we have changed the direction of dimension such that variance is maximum along that
## PCA will work only if you have Correlated data
## PCA fails where there are no correlation at all when samples are independent.
## In order to make data independent also it will not work because here we are concentrating only on the second order statistics
for i in range(m):
    r=p[i,:]
    for j in range(m):
        r2=YT2[:,j]
        Corr_matrix2[i][j]=np.dot(r,r2)/float(N)
print Corr_matrix2

