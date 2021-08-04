'''该函数直接运行就可以得到关联函数vs温度的图了，即文中figure7b，
会得到一张名为corre_vs_T.png 的图和一个名为 corre_vs_Tr1r2r3.txt 的txt文件
'''
import numpy as np
import matplotlib.pyplot as plt
import random


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
        for ki in range(self.r_dis):
            dic_thermal_t['corre_distance_{:}'.format(ki)]=[]
        for k in range(max1_nsweeps):
            self.sweep()     
     #       energy = np.sum(self.get_energy())/(2*self.width*self.length)
            #dic_thermal_t['energy'] += [energy]
            #dic_thermal_t['magnetism']+=[self.get_magcos()/(self.width*self.length)]
            #dic_thermal_t['vortex_density'] += [np.sum(self.vortex_density())/(self.width*self.length)]
            if (k>8000)  or k == max1_nsweeps-1:
                break

        
        
            #绘制能量,磁矩-步数图像
        after_steps=2**13
        for ki in range(int(after_steps)):
            self.sweep()
            for kj in range(self.r_dis):
                dic_thermal_t['corre_distance_{:}'.format(kj)]+=[self.correlation_func(kj)]
        self.corre_neigh1.append(np.sum(dic_thermal_t['corre_distance_{:}'.format(0)])/after_steps)#最近邻，第一近邻
        self.corre_neigh2.append(np.sum(dic_thermal_t['corre_distance_{:}'.format(2)])/after_steps)#第三近邻
        self.corre_neigh3.append(np.sum(dic_thermal_t['corre_distance_{:}'.format(6)])/after_steps)#第7近邻
            
            
        

    def annealing(self,T_init=2.5,T_final=0.1,nsteps = 20,show_equi=False):#同一个原始构型，不同温度下演化
        dic_thermal = {}
        dic_thermal['temperature']=np.linspace(T_init,T_final,nsteps)
        for T in dic_thermal['temperature']:
            #self.show()#用于检查每个温度下的初始构型是不是上一个温度的最终演化构型
            XY_triangular(width=8)
            self.equilibrate(temperature=T,equili_yes=False)
    
        
        
        return dic_thermal
    
        
    
    
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
