import numpy as np
import math
import random
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('bird_small.png')
#gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
## framing data matrix from the RGB pixels
[m,n,d]=img.shape
print m,n,d
Y=np.zeros([3,m*n])
kk=0
for i in range(m):
    for j in range(n):
        Y[:,kk]=img[i,j,:]
        kk=kk+1
print Y.shape
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x =Y[0,:]
y =Y[1,:]
z =Y[2,:]

ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
def k_means(k,Y,error):
    err_ite=0.2
    [m,N]=Y.shape
    print N
    Din=np.random.randn(m,k)*255
    
    Din_it=np.zeros([m,k],'float')
    while(err_ite>error):
        clusters=np.zeros([k,m,N])
        counts=np.ones([1,k],'int')
        Din_it=np.zeros([m,k])
        for i in range(N-1):
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
        print err_ite
    return Din_it,clusters,err_ite
##k_means clustering of color information using our k-means
k=5
error=0.01
[rr, clusters,err_final]=k_means(k,Y,error)
print rr
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
## framing back to another image
## check into which cluster it will fall and then take the centroid of that
clustered=np.zeros([m,n,3])
ii=0
for ix in range(m):
    for jx in range(n):
        matrix=[]
        for ki in range(k):       
            yy=np.sum((rr[:,ki]-Y[:,ii])**2)
            matrix.append(yy)
        index=np.argmin(matrix)
        ii=ii+1
        #print rr[:,index]
        clustered[ix,jx,:]=rr[:,index]
###dispaly
        ##print clustered
cv2.imshow('image',to_uint8(clustered))
cv2.waitKey(0)
cv2.destroyAllWindows()
