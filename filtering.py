#  
#  filtering.py
#  Loop_TRG
#  remove entanglement and CDL tensors
#
#  Copyright (C) 2018 Yue Zhengyuan, Liu Honghui and Zhang Wenqian. All rights reserved.
#  Article reference: Phys. Rev. Lett. 118, 110504 (2017)
#
import numpy as np
import tensorflow as tf
from itertools import product


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
    u, s, v = np.linalg.svd(L_inf)
    return u

def update_T(P1, P2, P3, P4, T):
    return T


