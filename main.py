#  
#  main.py
#  Loop_TRG
#  Main program for TRG calculation
#
#  Copyright (C) 2018 Yue Zhengyuan, Liu Honghui and Zhang Wenqian. All rights reserved.
#  Article reference: Phys. Rev. Lett. 118, 110504 (2017)
#

import numpy as np
import filtering as flt
import optimizing as opt
from itertools import product

temperature = 3

# assign the initial value of tensor TA, TB (2D Ising model)
ts_T_A0 = np.ones((2,2,2,2),dtype=complex)
ts_T_A0[0,1,0,1] = np.exp(-4/temperature)
ts_T_A0[1,0,1,0] = np.exp(-4/temperature)
ts_T_A0[0,0,0,0] = np.exp(4/temperature)
ts_T_A0[1,1,1,1] = np.exp(4/temperature)
ts_T_B0 = ts_T_A0.copy()

# the scaling is applied to the 4 -> 2 process
# tensor normalization constant
gamma_A0 = np.einsum('lulu', ts_T_A0)
gamma_B0 = np.einsum('lulu', ts_T_B0)
# normalized tensor before RG
ts_TN_A0 = ts_T_A0 / gamma_A0
ts_TN_B0 = ts_T_B0 / gamma_B0

for i in range(8):
    # normalized partition function for 4 sites
    part_ZN0 = np.einsum('ajkb,cbmj,mdci,kiad', ts_TN_A0,ts_TN_B0,ts_TN_A0,ts_TN_B0)
    # entanglement filtering
    ts_T_A1, ts_T_B1 = flt.filter(ts_TN_A0, ts_TN_B0, 1.0E-12)
    # loop optimize to find the new tensors
    ts_T_A1, ts_T_B1 = opt.loop_optimize((ts_T_A1,ts_T_B1), 16, 1E-6)
    # tensor normalization constant
    gamma_A1 = np.einsum('lulu', ts_T_A1)
    gamma_B1 = np.einsum('lulu', ts_T_B1)
    # normalized tensor after RG
    ts_TN_A1 = ts_T_A1 / gamma_A1
    ts_TN_B1 = ts_T_B1 / gamma_B1
    # normalized partition function for 2 sites
    part_ZN1 = np.einsum('dcba,badc', ts_TN_A1,ts_TN_B1)

    # scaling constant for normalized tensor (method 2)
    scale_fA = np.sqrt(part_ZN0/part_ZN1)
    # scale_fB = scale_fA

    ts_TN_A0 = ts_TN_A1.copy()
    ts_TN_B0 = ts_TN_B1.copy()

    print(i, scale_fA)
