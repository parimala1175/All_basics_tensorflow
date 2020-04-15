## Author K.Parimala
## IML_2nd Assignment
import numpy as np
import math
import random
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from matplotlib import cm
from scipy.stats import multivariate_normal
## this program is for fitting the gaussian Mixture Model for 2D data
## with Four mixtures having different means and with same covariance
dim=2
Model=4
print "Gaussian Mixture Model demo with 2D data and 4 models "
print "With Manual intilization "
## Input data generation and plotting
## I have taken few points just for demo purpose as it is taking lot of time
NUM_PTS=200
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
Y = np.concatenate((y1, y2,y3,y4), axis = 1)
plt.plot(Y[0,:], Y[1,:],'x')
plt.axis('equal')
plt.grid(True)
plt.xlabel('y1')
plt.ylabel('y2')
plt.hold(True)

### random intialization 
data_matrix=np.zeros([dim,NUM_PTS])
data_matrix=Y
## here I am manually intializing the alphas and means
## covariances too
alpha=np.ones([1,Model],dtype='f')*0.25
#mean=np.random.multivariate_normal([0, 0], [[1, 0],[0, 1]], 4).T
mean=np.zeros([dim,Model],dtype='f')
mean=[[0.6,0.5,-0.9,-0.7],[0.8,-0.9,-0.6,0.6]]

#mean=[[1,1],[1,-1],[-1, -1],[-1, 1]]
mean=np.array(mean)
Cova=np.zeros([dim,dim,Model],dtype='f')
#print Cova
Cova[:,:,0]=[[0.2,0],[0,0.2]]
Cova[:,:,1]=[[0.9,0],[0,0.2]]
Cova[:,:,2]=[[0.1,0],[0,0.3]]
Cova[:,:,3]=[[0.05,0],[0,0.2]]
## E step algorithm
## this computes the responsibilities 
def responsibility(x,mean,Cova,alpha,k,Model):
    mean_k=mean[:,k]
    cov_k=Cova[:,:,k]
    alpha_k=alpha[0][k]
    num=alpha_k*multivariate_normal.pdf(x,mean_k,cov_k)
    dd=0
    for j in range(Model):
        mean_j=mean[:,j]
        cov_j=Cova[:,:,j]
        alpha_j=alpha[0][j]
        dd=dd+alpha_j*multivariate_normal.pdf(x,mean_j,cov_j)
    responsibility=float(num)/float(dd)      
    return responsibility,dd
## M step in EM algorithm
## here I iterate the algorithm for means,COvariances
final_likelihood=52.75
likelihood_prev=0
K=Model          
N=4*NUM_PTS
X=np.zeros([2,N])
for i in range(N):
    X[:,i] = Y[:,random.randint(0,N-1)]
ite=0
## as the likelihood approaach is taking lot of time I am making it as iteration count just for demo purpose
## 
while((final_likelihood-likelihood_prev>0.09) or (ite<=20)):
    likelihood_prev=final_likelihood
    for k in range(K):
        mean_s=0
        cov=np.zeros([dim,dim])
        Nk=0
        res=[]
        for n in range(N):
            x=X[:,n]          
            [rr,dd]=responsibility(x,mean,Cova,alpha,k,Model)
            Nk=Nk+rr
            res.append(rr)
            mean_s=mean_s+rr*x       
        mean_k_est=mean_s/float(Nk)
        for n in range(N):
            x=X[:,n]
            x2=np.zeros([1,2])
            x2[0][0]=x[0]-mean_k_est[0]
            x2[0][1]=x[1]-mean_k_est[1]
            y=np.zeros([2,1])
            y[0][0]=x[0]-mean_k_est[0]
            y[1][0]=x[1]-mean_k_est[1]
            cov=cov+res[n]*np.dot(y,x2)
        cov_k_est=cov/float(Nk)
        alpha_k_est=Nk/N
        mean[:,k]=mean_k_est
        Cova[:,:,k]=cov_k_est
        alpha[0][k]=alpha_k_est        
    ### Likelihood evaluation
    final_liklihood=0
    for n in range(N):
        x=X[:,n]
        log=0
        for k in range(K):
            [rr,dd]=responsibility(x,mean,Cova,alpha,k,Model)
            log=log+math.log1p(dd)    
        final_likelihood= final_likelihood+log
    
    print final_likelihood
    ite=ite+1
    ### After converging the means and variances will be printed
print mean
print Cova
print alpha

   
    
    
    


        
    
        
