#  filtering.py
#  Loop_TRG
#  remove entanglement and CDL tensors
#
#  Copyright (C) 2018 Zhengyuan Yue, Honghui Liu and Wenqian Zhang. All rights reserved.
#  Article reference: Phys. Rev. Lett. 118, 110504 (2017)
#
import numpy as np
import tensorflow as tf
from itertools import product

# measure the difference between two tensors T1, T2
# using the sum of the squared difference
# between all corresponding elements in them resp. 
def tensor_error(T1, T2):
    error = 0.0
    dim = np.shape(T1)[0]
    for r, u, l, d in product(dim, dim, dim, dim): 
        error += (T1[r,u,l,d] - T2[r,u,l,d])**2
    return error

# find the matrix L_inf
def find_L_inf(T1, T2, T3, T4, epsilon):
    L_old = np.zeros([2,2,2,2])
    T = (T1, T2, T3, T4)
    error = np.inf
    while (error >= epsilon):
        for i in range(4):
            i = 1
            L_new = np.zeros([2,2,2,2])
    return L_new

# find the matrix R_inf
def find_R_inf(T1, T2, T3, T4, epsilon):
    R = np.zeros([2,2,2,2])
    return R

# find the projector PR, PL based on SVD of L_inf, R_inf
def find_P(L_inf, R_inf):
    TL_inf = tf.convert_to_tensor(L_inf)
    TR_inf = tf.convert_to_tensor(R_inf)
    s, u, v = tf.svd(TL_inf)
    return u

def update_T(P1, P2, P3, P4, T):
    return T


