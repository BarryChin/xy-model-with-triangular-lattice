# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 18:19:43 2021

@author: 找颗星星
"""
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.fftpack import fft
import sys
sys.path.append(r"C:\Users\找颗星星\Desktop\陈星森-020072910004-xymodel")#设置class的绝对路径
from xy_tri_class import XY_triangular

f=open("corre_vs_Tr1r2r3.txt","at") 

Tstep=30
Ti=0.02
Tf=3.0
dic_co={}
a = XY_triangular(width=10)
a.annealing(T_init=Ti,T_final=Tf,nsteps = Tstep,show_equi=False)

dic_co['temperature']=np.linspace(Ti,Tf,Tstep)

plt.figure(figsize=(5,4), dpi=500)
#yene = np.array(dic_thermal['Cv'])
#yene_smooth = make_interp_spline(x, yene)(x_smooth)
#plt.plot(x_smooth, yene_smooth)
plt.plot(dic_co['temperature'],a.corre_neigh1,label='nearest')
plt.plot(dic_co['temperature'],a.corre_neigh2,label='3rd neighbor')
plt.plot(dic_co['temperature'],a.corre_neigh3,label='7th neighbor')
plt.scatter(dic_co['temperature'],a.corre_neigh1)
plt.scatter(dic_co['temperature'],a.corre_neigh2)
plt.scatter(dic_co['temperature'],a.corre_neigh3)
plt.ylabel(r'correlation function')
plt.xlabel('T')
plt.legend()
plt.savefig('corre_vs_T.png')
plt.show()

print('nearest',file=f)
print(a.corre_neigh1,file=f)
print('3rd neighbor',file=f)
print(a.corre_neigh2,file=f)
print('7th neighbor',file=f)
print(a.corre_neigh3,file=f)


f.close()
