### third question estimates the parameters of distribution underlying for the data
## By the MLE estimation
## for the 5 distributions generated data based on the estimated parameters and found the mean square error
import numpy as np
import math
import random
from matplotlib import pyplot as plt
## assuming X is the input data
def binomial(X):
    N=len(X)
    p=sum(X)/float(N)
    return p;
def Poisson(X):
    N=len(X)
    gamma=sum(X)/float(N)
    return gamma;
def gaussian(X):
    N=len(X)
    mu=sum(X)/float(N)
    var=np.sqrt(sum((X-mu)**2)/float(N))
    return mu,var;
def exponential(X):
    N=len(X)
    gamma=float(N)/sum(X)
    return gamma
def laplacian(X):
   a=np.median(X)
   b=sum(abs(X-a))/float(N)
   return a,b
## given X and we select the distribution and estimate the parameters
n=10
p=0.6
N=1000
data1=np.random.binomial(n,p,[N,1])
p_est=binomial(data1)/n
gene_data=np.random.binomial(n,p_est,[N,1])
err=np.mean(np.subtract(data1,gene_data)**2)
## parameter estimation done for Binomial
print 'Binomial\n'
print 'error_found for binomial\n'
print err
print 'original parameters \n'
print p
print 'estimated parameters \n'
print p_est
## poisson
gamma=0.4
data2=np.random.poisson(gamma,[N,1])
gamma2=Poisson(data2)
gene_data=np.random.poisson(gamma2,[N,1])
err=np.mean(np.subtract(data2,gene_data)**2)
## parameter estimation done for poisson
print 'poisson\n'
print 'error_found for Poisson\n'
print err
print 'original parameters \n'
print gamma
print 'estimated parameters\n'
print gamma2
sigma=0.1;
mu=4;
data3=np.random.randn(N,1)*sigma+mu
mu2,sigma2=gaussian(data3)
gene_data=np.random.randn(N,1)*sigma2+mu2
err=np.mean(np.subtract(data3,gene_data)**2)
## parameter estimation done for Gaussian
print 'Gaussian\n'
print 'error_found for gaussian\n'
print err
print 'original parameters sigma , mean \n'
print sigma,mu
print 'estimated parameters sigma,mean\n '
print sigma2,mu2
a=1
b=2
data5=np.random.laplace(a,b,[N,1])
[a2,b2]=laplacian(data5)
gene_data=np.random.laplace(n,p_est,[N,1])
err=np.mean(np.subtract(data5,gene_data)**2)
print 'Laplacian\n'
print 'original parameters a and b\n'
print a,b
print 'estimated parameters a and b\n'
print a2,b2
print 'error_found for laplacian\n'
print err
## parameter estimation done for laplacian
lam1=0.9
data4=np.random.exponential(1/lam1,[N,1])
gamma2=exponential(data4)
gene_data=np.random.exponential(gamma2,[N,1])
err=np.mean(np.subtract(data4,gene_data)**2)
print 'Exponential\n'
print 'original parameters lamda\n '
print lam1
print 'estimated parameters lamda\n '
print gamma2
print 'error_found for laplacian\n'
print err

## parameter estimation done for exponential
