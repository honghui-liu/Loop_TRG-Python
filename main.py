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

beta = 1/4

# assign the initial value of tensor TA, TB (2D Ising model)
ts_TA = np.ones((2,2,2,2),dtype=complex)
ts_TA[0,1,0,1] = np.exp(-4*beta)
ts_TA[1,0,1,0] = np.exp(-4*beta)
ts_TA[0,0,0,0] = np.exp(4*beta)
ts_TA[1,1,1,1] = np.exp(4*beta)
ts_TB = ts_TA.copy()
result1 = np.einsum('ajkb,cbmj,mdci,kiad',ts_TA,ts_TB,ts_TA,ts_TB)

# entanglement filtering
ts_TA, ts_TB = flt.filter(ts_TA, ts_TB)
result2 = np.einsum('ajkb,cbmj,mdci,kiad',ts_TA,ts_TB,ts_TA,ts_TB)

# loop optimize
ts_TA, ts_TB = opt.loop_optimize((ts_TA,ts_TB), 16, 10E-10)
result3 = np.einsum('ajkb,cbmj,mdci,kiad',ts_TA,ts_TB,ts_TA,ts_TB)

print(result1,result2,result3)
