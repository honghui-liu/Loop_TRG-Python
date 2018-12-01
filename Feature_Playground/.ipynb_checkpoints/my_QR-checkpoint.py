# sef-defined QR decomposition function
# for array of dimention (2,2,2,2)

import numpy as np
import sys

# Check the dimention of input array
# (2,2,2,2)
def dimention_check(T):
    check_dimention = 0
    if (len(T.shape)!=4):
        print('input tensor:',T,'\nerror: Input array must be four dimentional')
        sys.exit(0)
    for i in range(4):
        check_dimention =check_dimention  + (T.shape[i]-2)**2
    if (check_dimention != 0):
        print('input tensor:',T,'\nerror: The dimention of the input tensor must be (2,2,2,2)')
        sys.exit(0)

# define QR decomposition of array with 
# dimention (2,2,2,2)
def four_D_QR(T):
    
    dimention_check(T)
    Q = np.ones([2,2,2,2])
    R = np.ones([2,2,2,2])
    
    for j in range(2):
        for k in range(2):
            Q[j,k] = np.linalg.qr(T[j,k])[0]
            R[j,k] = np.linalg.qr(T[j,k])[1]
    return Q, R

# define the product of (2,2,2,2) array
# checked that four_D_dot(Q,R) can give back
# the input tensor T
def four_D_dot(T1,T2):
    dimention_check(T1)
    dimention_check(T2)
    product = np.ones([2,2,2,2])
    for i in range(2):
        for j in range(2):
            product[i,j] = np.dot(T1[i,j],T2[i,j])
    return product

def getvalue(i):
    if (i==0):
        value = -1.0
    elif (i==1):
        value = 1.0
    return value

def gettensor(belta):
    T = np.ones([2,2,2,2])
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    T[i,j,k,l] = np.exp(belta*(getvalue(i)*getvalue(j) + getvalue(j)*getvalue(k) +
                                               getvalue(k)*getvalue(l) + getvalue(l)*getvalue(i)))
    return T