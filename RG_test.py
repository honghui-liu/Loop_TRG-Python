#  
#  filtering_test.py
#  Loop_TRG
#  
#  Copyright (C) 2018 Yue Zhengyuan, Liu Honghui and Zhang Wenqian. All rights reserved.
#  Article reference: Phys. Rev. Lett. 118, 110504 (2017)
#

import numpy as np
import filtering as flt
import optimizing as opt
from itertools import product


temperature = 2

# assign the initial value of tensor TA, TB (2D Ising model)
ts_T_A0 = np.ones((2,2,2,2),dtype=complex)
ts_T_A0[0,1,0,1] = np.exp(-4/temperature)
ts_T_A0[1,0,1,0] = np.exp(-4/temperature)
ts_T_A0[0,0,0,0] = np.exp(4/temperature)
ts_T_A0[1,1,1,1] = np.exp(4/temperature)
ts_T_B0 = ts_T_A0.copy()
# partition function for 4 sites
part_Z0 = np.einsum('ajkb,cbmj,mdci,kiad', ts_T_A0,ts_T_B0,ts_T_A0,ts_T_B0)

# RG: 4 -> 2
for i in range(3):
    # partition function for 4 sites
    part_Z0 = np.einsum('ajkb,cbmj,mdci,kiad', ts_T_A0,ts_T_B0,ts_T_A0,ts_T_B0)
    # entanglement filtering
    ts_T_A1, ts_T_B1 = flt.filter(ts_T_A0, ts_T_B0, 1.0E-12)
    # partition function for 4 sites
    part_Z0_flt = np.einsum('ajkb,cbmj,mdci,kiad', ts_T_A1,ts_T_B1,ts_T_A1,ts_T_B1)
    # loop optimize to find the new tensors
    ts_T_A1, ts_T_B1 = opt.loop_optimize((ts_T_A1,ts_T_B1), 16, 1E-10)
    # partition function for 2 sites
    part_Z1 = np.einsum('dcba,badc', ts_T_A1,ts_T_B1)
    print(i, part_Z0, part_Z1)
    ts_T_A0 = ts_T_A1.copy()
    ts_T_B0 = ts_T_B1.copy()

# RG: 8 -> 4
# for i in range(2):
#     # partition function for 8 sites
#     # part_Z0 = np.einsum('mnfe,pqhg,heib,fgaj,ijkl,abcd,clmq,kdpn',ts_T_A0,ts_T_A0,ts_T_B0,ts_T_B0,ts_T_A0,ts_T_A0,ts_T_B0,ts_T_B0)
#     # entanglement filtering
#     ts_T_A1, ts_T_B1 = flt.filter(ts_T_A0, ts_T_B0, 1.0E-12)
#     # loop optimize to find the new tensors
#     ts_T_A1, ts_T_B1 = opt.loop_optimize((ts_T_A1,ts_T_B1), 16, 10E-10)
#     # partition function for 2 sites
#     part_Z1 = np.einsum('ajkb,cbmj,mdci,kiad', ts_T_A1,ts_T_B1,ts_T_A1,ts_T_B1)
#     print(i, part_Z1)
#     ts_T_A0 = ts_T_A1.copy()
#     ts_T_B0 = ts_T_B1.copy()
    