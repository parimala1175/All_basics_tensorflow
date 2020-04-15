import numpy as np
import math
import random
from matplotlib import pyplot as plt
# First generate random numbers
# For illustration, we will generate 2-D points
# Specifically, four clusters from 2-D Gaussian distribution
NUM_PTS = 2000
mean1 = [1, 1]
cov1 = [[0.1,0],[0,0.1]]
y1 = np.random.multivariate_normal(mean1, cov1, NUM_PTS).T 

mean2 = [1, -1]
cov2 = [[0.1,0],[0,0.1]]
y2 = np.random.multivariate_normal(mean2, cov2, NUM_PTS).T 

mean3 = [-1, -1]
cov3 = [[0.1,0],[0,0.1]]
y3 = np.random.multivariate_normal(mean3, cov3, NUM_PTS).T 

mean4 = [-1, 1]
cov4 = [[0.1,0],[0,0.1]]
y4 = np.random.multivariate_normal(mean4, cov4, NUM_PTS).T 
Y = np.concatenate((y1, y2, y3, y4), axis = 1)
plt.ion()
plt.pause(0.05)
plt.plot(Y[0,:], Y[1,:],'x')
plt.axis('equal')
plt.grid(True)
plt.xlabel('y1')
plt.ylabel('y2')
plt.title('K-means demo')
def k_means(k,Y,error):
    err_ite=0.2
    [m,N]=Y.shape
    Din=np.random.multivariate_normal([0, 0], [[1, 0],[0, 1]], 4).T
    plt.plot(Din[0,:], Din[1,:],'o')
    plt.pause(1)
    Din_it=np.zeros([m,k],'float')
    while(err_ite>error):
        clusters=np.zeros([k,m,N])
        counts=np.zeros([1,k],'int')
        Din_it=np.zeros([m,k])
        for i in range(N):
            matrix=[]
            for ki in range(k):       
                yy=np.sum((Din[:,ki]-Y[:,i])**2)
                matrix.append(yy)
            index=np.argmin(matrix)
            Din_it[:,index]=Din_it[:,index]+Y[:,i]
            column_idx=counts[0][index]
            clusters[index,:,column_idx]=Y[:,i]
            counts[0][index]=column_idx+1
        for k2 in range(k):
            Din_it[:,k2]=Din_it[:,k2]/float(counts[0][k2])
        err_ite=np.sum((Din_it-Din)**2)
        Din=Din_it
        plt.plot(Din[0,:], Din[1,:],'o')
        plt.pause(1)
        print err_ite
    return Din_it,clusters,err_ite
### k_means clustering
k=4
error=0.01
[rr, clusters,err_final]=k_means(k,Y,error)
print err_final
