#这个只用来观察演化过程，输出三个状态图
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.fftpack import fft
import sys
sys.path.append(r"C:\Users\找颗星星\Desktop\陈星森-020072910004-xymodel")#需要设置class的绝对路径
from xy_tri_class import XY_triangular


'''
xy_system_2 = XY_triangular(width=10)
cool_dat=xy_system_2.annealing(T_init=4,T_final=0.02,nsteps = 10,show_equi=False)
'''        
        
xy_system_1 = XY_triangular(temperature  =0.02, width = 30)
xy_system_1.show()
print('Energy per spin:%.3f'%xy_system_1.energy)
xy_system_1.equilibrate(equili_yes=False,show=True,show_after=True)
xy_system_1.show()

