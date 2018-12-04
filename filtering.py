import numpy as np
import find_fixed_point as ffp

def get_one_projector(Left_fixed_point, Right_fixed_point, sv_cutoff):
    left = Left_fixed_point.copy()
    right = Right_fixed_point.copy()
    u, s, vh = np.linalg.svd(np.dot(left,right))
    
    # delete singular values that are smaller than sv_cutoff
    j = 0
    while j < len(s):
        if(s[j] < sv_cutoff):
            u = np.delete(u,j,axis = 1)
            vh = np.delete(vh,j,axis = 0)
            s = np.delete(s,j,axis = 0)
            j = j - 1
        else:
            j = j + 1

    # for i in range(length):
    #     if(s[j] < sv_cutoff):
    #         u = np.delete(u,i,axis = 1)
    #         vh = np.delete(vh,i,axis = 0)
    #         s = np.delete(s,i,axis = 0)
    #         j = j - 1
    #     else:
    #         j = j + 1   
            
    # get the square root and reciprocal of s
    modified_s = 1/np.sqrt(s)

    # modified_s = s
    # for i in range(len(s)):
    #     modified_s[i] = np.sqrt(modified_s[i])
    #     modified_s[i] = np.reciprocal(modified_s[i])
        
    # get all the matrices needed to compute projector    
    modified_s = np.diag(modified_s)
    uh = ffp.dagger(u)
    v = ffp.dagger(vh)
    
    right_projector = np.dot(np.dot(right,v),modified_s)
    left_projector  = np.dot(np.dot(modified_s,uh),left)
    return left_projector,right_projector

def filter(Tensor_A,Tensor_B, sv_cutoff):
    D = Tensor_A.shape[0]
    filter_tensor_A = Tensor_A.copy()
    filter_tensor_B = Tensor_B.copy()
    
    #get all the fixed point
    All_left_fixed_point = ffp.left_fixed_point(Tensor_A,Tensor_B,Tensor_A,Tensor_B,D)
    All_right_fixed_point = ffp.right_fixed_point(Tensor_A,Tensor_B,Tensor_A,Tensor_B,D)
    
    #get all projectors
    All_left_projector = [None]*4
    All_right_projector = [None]*4
    All_left_projector[0], All_right_projector[3] = get_one_projector(All_left_fixed_point[0],All_right_fixed_point[3],sv_cutoff)
    All_left_projector[1], All_right_projector[0] = get_one_projector(All_left_fixed_point[1],All_right_fixed_point[0],sv_cutoff)
    All_left_projector[2], All_right_projector[1] = get_one_projector(All_left_fixed_point[2],All_right_fixed_point[1],sv_cutoff)
    All_left_projector[3], All_right_projector[2] = get_one_projector(All_left_fixed_point[3],All_right_fixed_point[2],sv_cutoff)
    
    #contract projector with original tensor
    filter_tensor_A = np.einsum('mi,ijkl->mjkl',All_left_projector[0],filter_tensor_A)
    filter_tensor_A = np.einsum('ijkl,jm->imkl',filter_tensor_A,All_right_projector[2])
    filter_tensor_A = np.einsum('mk,ijkl->ijml',All_left_projector[2],filter_tensor_A)
    filter_tensor_A = np.einsum('ijkl,lm->ijkm',filter_tensor_A,All_right_projector[0])
    
    filter_tensor_B = np.einsum('ijkl,im->mjkl',filter_tensor_B,All_right_projector[1])
    filter_tensor_B = np.einsum('mj,ijkl->imkl',All_left_projector[1],filter_tensor_B)
    filter_tensor_B = np.einsum('ijkl,km->ijml',filter_tensor_B,All_right_projector[3])
    filter_tensor_B = np.einsum('ml,ijkl->ijkm',All_left_projector[3],filter_tensor_B)
    
    return filter_tensor_A, filter_tensor_B