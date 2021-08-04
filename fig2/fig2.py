#用于绘制不同温度下的关联函数随距离的变化
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import sys
sys.path.append(r"C:\Users\找颗星星\Desktop\陈星森-020072910004-xymodel")#需要设置class的绝对路径
from xy_tri_class import XY_triangular



f=open("corre_vs_rT.txt","at")    


dic = {}
temp_num=4
dic['temp']=list(np.linspace(0.02,3.0,temp_num))
#dic['Cv']=[]
a=XY_triangular(width=8)
#initial_system.show()
ei=el=1
x=np.arange(1,a.r_dis+1,1)
x_smooth = np.linspace(x.min(), x.max(), 30)
for i in range(1,temp_num+1):
    dic['corre_{:}'.format(i)]=np.zeros(a.r_dis)
plt.figure(figsize=(6,4), dpi=400)
for T in dic['temp']:
    a.equilibrate(temperature=T)
    print('energy=%.2f'%a.energy)
    #print(a.y_corre)
    for i in range(a.r_dis):
        dic['corre_{:}'.format(ei)][i] = a.y_corre[i]
    #print(dic['corre_{:}'.format(ei)])
    
    ei+=1

for T in dic['temp']:
    print('Temp={:*^10.3f}'.format(T),file=f)
    print(dic['corre_{:}'.format(el)],file=f)
    y=np.array(dic['corre_{:}'.format(el)]).reshape(a.r_dis)
    y_smooth = make_interp_spline(x, y)(x_smooth)
    plt.plot(x_smooth,y_smooth,label='T={:.3}'.format(T))
    plt.scatter(x,y)
    el+=1

plt.ylabel('correlation function ')
plt.xlabel('distance')
plt.legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0)
#plt.title('T=%.2f'%self.temperature)
plt.savefig('corr_vs_distan_temp.jpg')
plt.show()


f.close()