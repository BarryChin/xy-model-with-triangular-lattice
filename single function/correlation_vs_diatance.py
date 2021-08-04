'''这是绘制关联函数vs距离图像，生成两个文件corr_vs_distan.jpg和corre_vs_r.txt
'''
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.interpolate import make_interp_spline



class XY_triangular():
    def __init__(self,temperature = 3,width=10):
        self.width = width#二维正方形宽度
        if self.width % 2 == 1:
            self.length=self.width+1
        else:
            self.length=self.width
        self.num_spins = (self.width*2)*self.length#总的粒子数,由于我打算每个自旋间隔一个空位来形成三角形，所以宽度先乘以2才能得到想要的数量
        L,N = self.width*2,self.num_spins
        self.nbr = {i : (  (i//L)*L+(i+2)%L, (i//L)*L+ (i-2)%L, (((i-L)//L)*L)%N+((i-L)-1)%L,  (((i-L)//L)*L)%N+((i-L)+1)%L,(((i+L)//L)*L)%N+((i+L)-1)%L,  (((i+L)//L)*L)%N+((i+L)+1)%L ) for i in list(range(N))}
        self.spin_config = np.zeros(self.num_spins)
        self.spin_index=[]
        for i in range(N):#用于找出构建自旋的点
            if (((i//L+1) % 2) == 0) & ((i % 2)==0):
                self.spin_index.append(i)
            elif ((i//L+1) % 2 == 1) & ((i % 2)==1):
                self.spin_index.append(i)
        for i in range(int(N/2)):
            self.spin_config[self.spin_index[i]]=list(np.zeros(int(N/2)))[i]#list(np.random.random(int(N/2))*2*pi)[i]#构造xy model的自旋取向random.random取值为0到1
        
        for i in range(self.num_spins):#让不用来构建自旋的点都为负数
            if i in self.spin_index:
                continue
            else:
                self.spin_config[i]=-2
        self.temperature = temperature
        self.energy = np.sum(self.get_energy())/(2*self.width*self.length)#求单个能量
        self.magnetism = []
        self.totalmag=[]
        self.Cv = []
        self.mag_suscept=[]#磁化率
        self.r_dis=10 #关联函数长度
        self.y_corre=np.zeros(self.r_dis)
        self.corre_neigh1=[]
        self.corre_neigh2=[]
        self.corre_neigh3=[]
        

    
    def sweep(self):
        beta = 1.0 / self.temperature
        for idx in random.sample(self.spin_index,len(self.spin_index)):
            energy_i = -sum( np.cos( self.spin_config[idx] - self.spin_config[n] ) for n in self.nbr[idx] ) 
            dtheta = np.random.uniform(0*np.pi,2*np.pi)
            spin_temp = self.spin_config[idx] + dtheta#取向变化
            energy_f = -sum(  np.cos(  spin_temp - self.spin_config[n]  ) for n in self.nbr[idx]  ) 
            delta_E = energy_f - energy_i
            if delta_E < 0 or np.random.uniform(0.0, 1.0) < np.exp(-beta * delta_E):#判断翻转有不有效果，无论最后翻不翻转结果都视为演化一次
                self.spin_config[idx] += dtheta
                if self.spin_config[idx] > 2*np.pi:
                    self.spin_config[idx] -= 2*np.pi


    def get_energy(self):
        energy_=np.zeros(np.shape(self.spin_index))#shape返回一个数组
        idx = 0
        for spin in self.spin_index: #calculate energy per spin每一个自旋的能量
            energy_[idx] = -sum(np.cos(self.spin_config[spin]-self.spin_config[n]) for n in self.nbr[spin])#nearst neighbor of kth spin
            idx +=1
        return energy_
    
    def get_magcos(self):
        magnetism_ = 0#求磁矩序参量
        for spin in self.spin_index:
            magnetism_ += (np.cos(self.spin_config[spin]))#+np.sin(self.spin_config[spin])   )
        return magnetism_
    def get_magsin(self):
        magnetism_ = 0#求磁矩序参量
        for spin in self.spin_index:
            magnetism_ += (np.sin(self.spin_config[spin]))#+np.sin(self.spin_config[spin])   )
        return magnetism_
    
    
    def vortex_density(self):
        vortex_=np.zeros(np.shape(self.spin_index))#shape返回一个数组
        idx = 0
        for spin in self.spin_index: 
            vortex_[idx] = sum((self.spin_config[spin]-self.spin_config[n]) for n in self.nbr[spin])/(2*np.pi)#nearst neighbor of kth spin
            idx +=1
        return vortex_
    
    def correlation_func(self,ri):
        L,N = self.width*2,self.num_spins
        corre=0.0
        for spin in self.spin_index:
            a1=[]
            a1.append( (spin//L)*L+(spin+2*(ri+1))%L)
            a1.append((spin//L)*L+ (spin-2*(ri+1))%L)
            a1.append((((spin-(ri+1)*L)//L)*L)%N+((spin-(ri+1)*L)-(ri+1))%L)
            a1.append((((spin-(ri+1)*L)//L)*L)%N+((spin-(ri+1)*L)+1+ri)%L)
            a1.append((((spin+(ri+1)*L)//L)*L)%N+((spin+(ri+1)*L)-ri-1)%L)
            a1.append((((spin+(ri+1)*L)//L)*L)%N+((spin+(ri+1)*L)+ri+1)%L)
            for i in range(len(a1)):
                corre+=np.cos(self.spin_config[spin]-self.spin_config[a1[i]] )
        return corre/(len(a1)*self.width*self.length)
        

    def equilibrate(self,max1_nsweeps=int(1e4),temperature=None,show = False,show_after=False,equili_yes=True):
        if temperature != None:
            self.temperature = temperature
        dic_thermal_t = {}
        dic_thermal_t['energy']=[]
        dic_thermal_t['magnetism']=[]
        dic_thermal_t['totalmag']=[]
        #dic_thermal_t['vortex_density']=[]
        for ki in range(self.r_dis):
            dic_thermal_t['corre_distance_{:}'.format(ki)]=[]
        for k in range(max1_nsweeps):
            self.sweep()     
            if (k>8000)  or k == max1_nsweeps-1:
                break

        
        after_steps=2**10
        for ki in range(int(after_steps)):
            self.sweep()
            for kj in range(self.r_dis):
                dic_thermal_t['corre_distance_{:}'.format(kj)]+=[self.correlation_func(kj)]
        for ki in range(self.r_dis):
            self.y_corre[ki] = np.sum(dic_thermal_t['corre_distance_{:}'.format(ki)])/after_steps

            
            
                
    

    

f=open("corre_vs_r.txt","at")    


dic = {}
temp_num=4
dic['temp']=list(np.linspace(0.02,3.0,temp_num))
#dic['Cv']=[]
a=XY_triangular(width=20)#维度需大于10
#initial_system.show()
ei=el=1
x=np.arange(1,a.r_dis+1,1)
x_smooth = np.linspace(x.min(), x.max(), 30)
for i in range(1,temp_num+1):
    dic['corre_{:}'.format(i)]=np.zeros(a.r_dis)
plt.figure(figsize=(6,4), dpi=400)
for T in dic['temp']:
    a.equilibrate(temperature=T)
    print('down!!!')
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
plt.savefig('corr_vs_distan.jpg')
plt.show()


f.close()