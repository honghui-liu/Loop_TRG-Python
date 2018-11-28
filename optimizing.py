#  optimizing.py
#  Loop_TRG
#  perform SVD to transform square to octagon and optimize it
#
#  Copyright (C) 2018 Zhengyuan Yue, Honghui Liu and Wenqian Zhang. All rights reserved.
#  Article reference: Phys. Rev. Lett. 118, 110504 (2017)
#

import numpy as np
import tensorflow as tf
import filtering as ft

# get the initial value of the 8 S's using SVD
# T: the tuple of tensors (TA, TB)
# (should be already converted to TensorFlow tensor)
def init_S(T):
    # decompose TA = T[0]
    sa, UA, VA = tf.svd(T[0], full_matrices=True)

    # decompose TB = T[1]
    sb, UB, VB = tf.svd(T[1], full_matrices=True)

# calculate the tensor N_i (i = 0 ~ 7)
# S: the tuple of tensors S[j] (j = 0 ~ 7)
# (should be already converted to TensorFlow tensor)
def tensor_N(i, S):
    num = len(S) - 1
    # j takes values i+1, ..., len(S)-1, 0, ..., i-1
    adj_S = tf.linalg.adjoint(S)
    for j in range(i+1, i+num-2):
        # contract (S[j])+ and S[j] first
        A = tf.einsum('ijk,lmk->ijlm', adj_S[j], S[j])
        if j == i + 1:
            N = A
        elif j > i + 1:
            if j <= num - 1:
                N = tf.einsum('ijlm,jpmq->iplq', N, A[j])
            elif j >= len(S):
                N = tf.einsum('ijlm,jpmq->iplq', N, A[j - num])
    return N

# calculate the tensor W_i
# S: the tuple of tensors S[j] (j = 0 ~ 7)
# T: the tuple of tensors (TA, TB)
# (should be already converted to TensorFlow tensor)
def tensor_W(i, S, T):
    # tensors in pair
    # 0: S[0],S[1],T[0]; 1: S[2],S[3],T[1], ...
    # j: S[2j],S[2j+1],T[ab]
    pair = int(i / 2)
    # ab = 0 -> use TA = T[0]; ab = 1 -> use TB = T[1]
    ab = pair % 2
    adj_S = tf.linalg.adjoint(S)

    if i % 2 == 0:      # W starts with C
        C = tf.einsum('aim,nmpq->ainpq', adj_S[i+1], T[ab])
        # for the other more "regular" tensors A
        for p in range(pair, pair + 4):
            if p <= 3:
                j = p
            elif p >= 4:
                j = p - 4

            if j == pair:
                W = C
            elif j == pair + 3:
                A = tf.einsum('ilm,ljn,mnpq->ijpq', adj_S[2*j], adj_S[2*j+1], T[int(j/2)%2])
                W = tf.einsum('ainpj,iljp->aln', W, A)
            else: 
                A = tf.einsum('ilm,ljn,mnpq->ijpq', adj_S[2*j], adj_S[2*j+1], T[int(j/2)%2])
                W = tf.einsum('ainpj,icjd->acnpd', W, A)

    elif i % 2 != 0:    # W ends with C
        C = tf.einsum('ilm,mnpq->ilnpq', tf.linalg.adjoint(S[i-1]), T[ab])
        # for the other more "regular" tensors A
        for p in range(pair, pair + 4):
            if p <= 3:
                j = p
            elif p >= 4:
                j = p - 4

            if j == pair:
                A = tf.einsum('ilm,ljn,mnpq->ijpq', adj_S[2*j], adj_S[2*j+1], T[int(j/2)%2])
                W = A
            elif j == pair + 3:
                W = tf.einsum('aibp,ilnpb->aln', W, C)
            else:
                A = tf.einsum('ilm,ljn,mnpq->ijpq', adj_S[2*j], adj_S[2*j+1], T[int(j/2)%2])
                W = tf.einsum('acbd,csdr->asbr', W, A)
    return W