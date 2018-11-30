#  optimizing.py
#  Loop_TRG
#  perform SVD to transform square to octagon and optimize it
#
#  Copyright (C) 2018 Yue Zhengyuan, Liu Honghui and Zhang Wenqian. All rights reserved.
#  Article reference: Phys. Rev. Lett. 118, 110504 (2017)
#

import numpy as np
from itertools import product
import filtering as ft

# get the initial value of the 8 S's using SVD
# ts_T: the tuple of tensors (TA, TB)
# d_cut: the upper limit of number of singular values to be kept
# return: the initial 8 S's

def init_S(ts_T, d_cut):
    d = ts_T.shape[0]
    d2 = d**2
    dc = min(d2, d_cut)
    mat = []
    # convert T[0]/T[1] to (d^2)x(d^2) matrix mat[0], mat[1]
    mat.append(ts_T[0].reshape((d2,d2))) 
    mat.append((np.einsum('lijk->ijkl',ts_T[1])).reshape((d2,d2)))
    # do svd for mat
    ts_Result = []
    for i in range(2):
        mat_U, s, mat_V = np.linalg.svd(mat[i], full_matrices=True)
        # keep the largest Dc singular values
        s = s[0:dc]
        diag_s = np.diag(s)
        s1 = np.sqrt(diag_s)
        s2 = np.sqrt(diag_s)
        # keep the corresponding (first dc) rows/columns in u/v
        mat_U = mat_U[:,0:dc]
        mat_V = mat_V[0:dc,:]
        # find the decomposition
        mat1 = np.matmul(mat_U,s1)
        mat2 = np.matmul(s2,mat_V)
        # convert the resulted matrix to dxdxdc tensor
        ts_S1 = mat1.reshape((d,d,dc))
        ts_S2 = mat2.reshape((d,d,dc))
        ts_Result.append(ts_S1)
        ts_Result.append(ts_S2)
    
    ts_Result.append(ts_Result[1])
    ts_Result.append(ts_Result[0])
    ts_Result.append(ts_Result[3])
    ts_Result.append(ts_Result[2])
    # elements in result:
    # S1, S2, S3, S4, S2, S1, S4, S3
    return tuple(ts_Result)

# calculate the tensor N_i (i = 0 ~ 7)
# ts_S: the tuple of tensors S[k] (k = 0 ~ 7)
# return: ts_N_i

def tensor_N(i, ts_S):
    num = len(ts_S)     # should be 8
    ts_S_conj = np.conj(ts_S)
    for j in range(i+1, i+num-1):
        # k takes values i+1, ..., len(S)-1, 0, ..., i-1
        if j <= num - 1:
            k = j
        elif j >= len(ts_S):
            k = j - len(ts_S)
        # contract (S[k])+ and S[k] first to find a tensor ts_A
        ts_A = np.einsum('icj,sct->ijst', ts_S_conj[k], ts_S[k])
        if j == i + 1:
            ts_N = ts_A
        else: 
            ts_N = np.einsum('abcd,bedf->aecf', ts_N, ts_A)
    return ts_N

# calculate the tensor W_i
# ts_S: the tuple of tensors S[j] (j = 0 ~ 7)
# ts_T: the tuple of tensors (TA, TB)
# return: ts_W_i

def tensor_W(i, ts_S, ts_T):
    # tensors in pair
    # 0: S[0],S[1],T[0]; 1: S[2],S[3],T[1];
    # 2: S[4],S[5],T[0]; 3: S[6],S[7],T[1];
    # --> j: S[2j],S[2j+1],T[ab]
    pair = int(i / 2)
    # ab = 0 -> use TA = T[0]; ab = 1 -> use TB = T[1]
    ab = pair % 2
    ts_S_conj = np.conj(ts_S)

    if i % 2 == 0:
        ts_C = np.einsum('bed,fceg->bdcfg', ts_S_conj[i+1], ts_T[ab])
        for p in range(pair, pair + 4):
            # j takes the value p, p+1, ... 3, 0, ..., p-1
            if p <= 3:
                j = p
            elif p >= 4:
                j = p - 4

            if j == pair:       # W starts with C
                ts_W = ts_C
            elif j == pair + 3: # connect the last tensor (B)
                ts_B = np.einsum('dpm,mqn,gpqr->dngr', ts_S_conj[2*j], ts_S_conj[2*j+1], ts_T[int(j/2)%2])
                ts_W = np.einsum('bncfr,narf->bac', ts_W, ts_B)
            else: 
                ts_B = np.einsum('dpm,mqn,gpqr->dngr', ts_S_conj[2*j], ts_S_conj[2*j+1], ts_T[int(j/2)%2])
                ts_W = np.einsum('bdcfg,dngr->bncfr', ts_W, ts_B)

    elif i % 2 != 0:
        ts_C = np.einsum('dae,fecg->dacfg', np.conj(ts_S[i-1]), ts_T[ab])
        for p in range(pair, pair + 4):
            if p <= 3:
                j = p
            elif p >= 4:
                j = p - 4

            if j == pair:       # W starts with B
                ts_B = np.einsum('bpm,mqn,gpqr->bngr', ts_S_conj[2*j], ts_S_conj[2*j+1], ts_T[int(j/2)%2])
                ts_W = ts_B
            elif j == pair + 3: # connect the last tensor (C)
                ts_W = np.einsum('ndgf,dacfg->aln', ts_W, ts_C)
            else:
                ts_B = np.einsum('bpm,mqn,gpqr->bngr', ts_S_conj[2*j], ts_S_conj[2*j+1], ts_T[int(j/2)%2])
                ts_W = np.einsum('abcd,bedf->aecf', ts_W, ts_B)
    return ts_W

# solve the equation (N_i)(S_i) = (W_i) for (S_i)
# return: the optimized S_i = (N_i)^(-1) * (W_i)
def optimize(ts_N, ts_W):
    # convert tensor N_(abcd) to matrix N_(ab,cd)
    mat_N = ts_N.reshape((ts_N.shape[0]*ts_N.shape[1], ts_N.shape[2]*ts_N.shape[3]))
    # convert tensor W_(abe) to matrix W_(ab,e)
    mat_W = ts_W.shape((ts_W.shape[0]*ts_W.shape[1], ts_W.shape[2]))
    # find matrix S'_(cd,e)
    mat_S = np.dot(np.inv(mat_N), mat_W)
    # convert matrix S' to tensor S'_(cde)
    ts_S = mat_S.reshape((ts_N.shape[2]*ts_N.shape[3], ts_W.shape[2]))
    # find the required S using S_(dec) = S'_(cde)
    ts_S = np.einsum('cde->dec', ts_S)
    return ts_S

# loop-optimize the 8 tensors ts_S[i] (i = 0 ~ 7)
# until the "distance" between the tensors ts_S[i] 
# of two optimizations is smaller than error_limit
# return: optimized ts_S
def loop_optimize(ts_T, ts_S_original, error_limit):
    error = np.inf
    ts_S_old = list(ts_S_original)
    ts_S_new = list(ts_S_original)
    num = len(ts_S_original)    # should be 8
    while (error > error_limit):
        for i in range(num):
            ts_N = tensor_N(i, ts_S_old)
            ts_W = tensor_W(i, ts_S_old, ts_T)
            ts_S_new[i] = optimize(ts_N, ts_W)
        error = np.linalg.norm(ts_S_new - ts_S_old)
        ts_S_old = ts_S_new.copy()
    return ts_S_new