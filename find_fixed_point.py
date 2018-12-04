import numpy as np
from itertools import product

def contract_and_qr(mat_R,ts_T,D):
    contract = np.einsum('ab,bcde->acde',mat_R,ts_T)
    matrix = contract.reshape((D*D*D,D))
    q, r = np.linalg.qr(matrix)
    return r

def rearrange_indice(T1,T2,T3,T4):
    re_T1 = np.einsum('abcd->abcd',T1)
    re_T2 = np.einsum('edfg->dfge',T2)
    re_T3 = np.einsum('jkei->eijk',T3)
    re_T4 = np.einsum('mnak->kmna',T4)
    ts_rearranged = (re_T1,re_T2,re_T3,re_T4)
    return ts_rearranged

def left_one_circle(ts_rearranged,L_old,D):
    old_L = L_old.copy()
    for i in range(4):
        new_L = contract_and_qr(old_L,ts_rearranged[i],D)
        old_L = new_L
    new_L = new_L/new_L.max()
    return new_L

def find_error(mat1,mat2,D):
    error = 0.0
    for i in range(D):
        for j in range(D):
            error = error + (mat1[i,j]-mat2[i,j])**2
    return error

def left_fixed_point(T1,T2,T3,T4,D):
    ini_fixed_point = np.ones([D,D])
    L_all = [ini_fixed_point,ini_fixed_point,ini_fixed_point,ini_fixed_point]
    
    for i in range(4):        
        ts_original = (T1,T2,T3,T4)
        epsilon = 1.e-10
        error = 1.0
        L_old = np.ones([D,D])
        L_new = np.ones([D,D])
        iteration = 0

        ts_rearranged = rearrange_indice(T1,T2,T3,T4)
        ts_rearranged_with_i = (ts_rearranged[(0+i)%4],ts_rearranged[(1+i)%4],
                                ts_rearranged[(2+i)%4],ts_rearranged[(3+i)%4])

        while(error > epsilon and iteration < 30):
            L_new = left_one_circle(ts_rearranged_with_i,L_old,D)
            error = find_error(L_old, L_new, D)
            L_old = L_new
            iteration = iteration + 1
        L_all[i] = L_new
        
    All_left_points = (L_all[0],L_all[1],L_all[2],L_all[3])
    
    return All_left_points

def contract_and_lq(l,tensor,D):
    contract = np.einsum('abcd,de->abce',tensor,l)
    matrix = contract.reshape((D,D*D*D))
    matrix = matrix.transpose()
    q, r = np.linalg.qr(matrix.conjugate())
    r = r.transpose()
    r = r.conjugate()
    return r

def right_one_circle(ts_rearranged,R_old,D):
    old_R = R_old.copy()
    for i in range(4):
        new_R = contract_and_lq(old_R,ts_rearranged[-i],D)
        old_R = new_R
    new_R = new_R/new_R.max()
    return new_R

def right_fixed_point(T1,T2,T3,T4,D):
    ini_fixed_point = np.ones([D,D])
    R_all = [ini_fixed_point,ini_fixed_point,ini_fixed_point,ini_fixed_point]
    
    for i in range(4):        
        ts_original = (T1,T2,T3,T4)
        epsilon = 1.e-10
        error = 1.0
        R_old = np.ones([D,D])
        R_new = np.ones([D,D])
        iteration = 0

        ts_rearranged = rearrange_indice(T1,T2,T3,T4)
        ts_rearranged_with_i = (ts_rearranged[(0+i)%4],ts_rearranged[(1+i)%4],
                                ts_rearranged[(2+i)%4],ts_rearranged[(3+i)%4])
        while(error > epsilon and iteration < 50):
            R_new = right_one_circle(ts_rearranged_with_i,R_old,D)
            error = find_error(R_old, R_new, D)
            R_old = R_new
            iteration = iteration + 1
        R_all[i] = R_new
        
    All_right_points = (R_all[0],R_all[1],R_all[2],R_all[3])
    return All_right_points

def dagger(matrix):
    in_matrix = matrix.copy()
    out_matrix = in_matrix.transpose()
    out_matrix = out_matrix.conjugate()
    return out_matrix

def getvalue(i):
    if (i==0):
        value = -1.0
    elif (i==1):
        value = 1.0
    return value

def gettensor(beta):
    T = np.ones([2,2,2,2])
    for i,j,k,l in product(range(2), repeat=4):
        T[i,j,k,l] = np.exp(beta*(getvalue(i)*getvalue(j) + getvalue(j)*getvalue(k) + getvalue(k)*getvalue(l) + getvalue(l)*getvalue(i)))
    return T