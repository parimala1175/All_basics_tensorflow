import numpy as np
import math
import random
import cv2
from matplotlib import pyplot as plt
from scipy import linalg
img = cv2.imread('Lenna.png')
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#framing data matrix from the RGB pixels
[m,n]=gray_image.shape
patch=8
Y=np.zeros([(patch**2),int(m*n/(patch**2))])
kk=0
for i in range(0,m-patch,patch):
    for j in range(0,n-patch,patch):
        f=gray_image[i:i+patch,j:j+patch]
        d=np.reshape(f,patch**2)
        Y[:,kk]=np.transpose(d)
        kk=kk+1
def PCA(Y):
    print Y.shape
    [m,N]=Y.shape
    Corr_matrix=np.zeros([m,m],'float')
    YT=np.transpose(Y)
    for i in range(m):
        r=Y[i,:]
        for j in range(m):
            r2=YT[:,j]
            Corr_matrix[i][j]=np.dot(r,r2)/float(N)
    ## Eigen value Decomposition
    D,V=linalg.eig(Corr_matrix)
    Transform=np.transpose(V)
    W=np.dot(V,Y)
    return W,Transform;
def to_uint8( data ) :
    # maximum pixe
    latch = np.zeros_like( data )
    latch[:] = 255
    # minimum pixel
    zeros = np.zeros_like( data )
    # unrolled to illustrate steps
    d = np.maximum( zeros, data )
    d = np.minimum( latch, d )
    # cast to uint8
    return np.asarray( d, dtype="uint8" )
k=4
error=0.01
p,transform=PCA(Y)
## we just validate the output p by calculating the covariance Matrix
ff=np.cov(p)
#print ff
py=np.transpose(transform)
rr3=np.dot(transform,p)
## framing back to image with the transformed W and the transformed Matrix
kk=0
gray_image2=np.zeros([m,n])
for i in range(0,m-patch,patch):
    for j in range(0,n-patch,patch):
        f=rr3[:,kk]
        kk=kk+1        
        d=np.reshape(f,(patch,patch))
        gray_image2[i:i+patch,j:j+patch]=d
cv2.imshow('image',to_uint8(gray_image2))
cv2.waitKey(0)
cv2.destroyAllWindows()
