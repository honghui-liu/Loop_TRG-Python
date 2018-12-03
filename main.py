#  
#  main.py
#  Loop_TRG
#  main program for the TRG calculation of the partition function
#
#  Copyright (C) 2018 Yue Zhengyuan, Liu Honghui and Zhang Wenqian. All rights reserved.
#  Article reference: Phys. Rev. Lett. 118, 110504 (2017)
#

import numpy as np
import filtering as flt
import optimizing as opt
from itertools import product

beta = 1/4

# assign the initial value of tensor TA, TB (2D Ising model)
ts_TA = np.ones((2,2,2,2),dtype=complex)
# ts_TA[0,1,0,1] = np.exp(-4*beta)
# ts_TA[1,0,1,0] = np.exp(-4*beta)
# ts_TA[0,0,0,0] = np.exp(4*beta)
# ts_TA[1,1,1,1] = np.exp(4*beta)
for l,u,r,d in product(range(2), repeat=4):
    ts_TA[l,u,r,d] = (1+(2*l-1)*(2*u-1)*(2*r-1)*(2*d-1))/2 * np.exp(beta*(l+u+r+d-2))
ts_TB = ts_TA.copy()

# partition function for 8 sites
# print(np.einsum('mnfe,pqhg,heib,fgaj,ijkl,abcd,clmq,kdpn',ts_TA,ts_TA,ts_TB,ts_TB,ts_TA,ts_TA,ts_TB,ts_TB))

for i in range(1,3):
    # entanglement filtering
    ts_TA, ts_TB = flt.filter(ts_TA, ts_TB)
    # loop optimize
    ts_TA, ts_TB = opt.loop_optimize((ts_TA,ts_TB), 16, 10E-5)
    partition_Z = np.einsum('ajkb,cbmj,mdci,kiad', ts_TA,ts_TB,ts_TA,ts_TB)
    print(i, partition_Z)
