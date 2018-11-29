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

# calculate the adjoint of one tensor
# transpose each sub-matrix separately
def adjoint(tensor):
    dim = tensor.ndim
    temp = np.swapaxes(tensor, dim - 2, dim - 1)
    temp = np.conj(temp)
    return temp

# convert 2x2x2x2 tensor to 4x4 matrix
def to_matrix(tensor):
    matrix = np.zeros((4,4))
    for l,i in product(range(2),repeat=2):
        matrix[2*l:2*l+2, 2*i:2*i+2] = tensor[l,i,:,:]
    return matrix

# convert 2x4 (4x2) matrix to 2x2x2 tensor
def to_tensor(matrix):
    tensor = np.zeros((2,2,2))
    return tensor

# get the initial value of the 8 S's using SVD
# T: the tuple of tensors (TA, TB)

def init_S(T):
    # convert T[0]/T[1] to 4x4 matrix mat[0], mat[1]
    mat = to_matrix(T)
    result = []
    # do svd for mat
    for i in range(2):
        u, s, vH = np.linalg.svd(mat[i], full_matrices=True)
        # keep the largest (first 2) singular values
        s = s[0:2]
        diag_s = np.diag(s)
        s1 = np.sqrt(diag_s)
        s2 = np.sqrt(diag_s)
        # keep the corresponding (first 2) rows/columns in u/vH
        u = u[:,0:2]
        vH = vH[0:2,:]
        # find the decomposition
        temp1 = np.matmul(u,s1)
        temp2 = np.matmul(s2,vH)
        # convert the resulted matrix temp1/2 to 2x2x2 tensor s_tensor1/2
        s_tensor1 = to_tensor(temp1)
        s_tensor2 = to_tensor(temp2)
        result.append(s_tensor1)
        result.append(s_tensor2)
    
    # repeat the list "result"
    result_len = len(result)
    for i in range(result_len):
        result.append(result[i])
    return result

# calculate the tensor N_i (i = 0 ~ 7)
# S: the tuple of tensors S[j] (j = 0 ~ 7)

def tensor_N(i, S):
    num = len(S) - 1
    # j takes values i+1, ..., len(S)-1, 0, ..., i-1
    adj_S = adjoint(S)
    for j in range(i+1, i+num-2):
        # contract (S[j])+ and S[j] first
        A = np.einsum('ijk,klm->ijlm', adj_S[j], S[j])
        if j == i + 1:
            N = A
        elif j > i + 1:
            if j <= num - 1:
                N = np.einsum('iljm,jmpq->ilpq', N, A[j])
            elif j >= len(S):
                N = np.einsum('iljm,jmpq->ilpq', N, A[j - num])
    return N

# calculate the tensor W_i
# S: the tuple of tensors S[j] (j = 0 ~ 7)
# T: the tuple of tensors (TA, TB)

def tensor_W(i, S, T):
    # tensors in pair
    # 0: S[0],S[1],T[0]; 1: S[2],S[3],T[1], ...
    # j: S[2j],S[2j+1],T[ab]
    pair = int(i / 2)
    # ab = 0 -> use TA = T[0]; ab = 1 -> use TB = T[1]
    ab = pair % 2
    adj_S = adjoint(S)

    if i % 2 == 0:      # W starts with C
        C = np.einsum('aim,nmpq->ainpq', adj_S[i+1], T[ab])
        # for the other more "regular" tensors A
        for p in range(pair, pair + 4):
            if p <= 3:
                j = p
            elif p >= 4:
                j = p - 4

            if j == pair:
                W = C
            elif j == pair + 3:
                A = np.einsum('ilm,ljn,mnpq->ijpq', adj_S[2*j], adj_S[2*j+1], T[int(j/2)%2])
                W = np.einsum('ainpj,iljp->aln', W, A)
            else: 
                A = np.einsum('ilm,ljn,mnpq->ijpq', adj_S[2*j], adj_S[2*j+1], T[int(j/2)%2])
                W = np.einsum('ainpj,icjd->acnpd', W, A)

    elif i % 2 != 0:    # W ends with C
        C = np.einsum('ilm,mnpq->ilnpq', adjoint(S[i-1]), T[ab])
        # for the other more "regular" tensors A
        for p in range(pair, pair + 4):
            if p <= 3:
                j = p
            elif p >= 4:
                j = p - 4

            if j == pair:
                A = np.einsum('ilm,ljn,mnpq->ijpq', adj_S[2*j], adj_S[2*j+1], T[int(j/2)%2])
                W = A
            elif j == pair + 3:
                W = np.einsum('aibp,ilnpb->aln', W, C)
            else:
                A = np.einsum('ilm,ljn,mnpq->ijpq', adj_S[2*j], adj_S[2*j+1], T[int(j/2)%2])
                W = np.einsum('acbd,csdr->asbr', W, A)
    return W

# loop-optimize the 8 matrices S[i] (i = 0 ~ 7)
# until the "distance" between the matrice S[i] 
# of two optimizations is smaller than error
def optimize(S, error):
    np.transpose(S)
    return S