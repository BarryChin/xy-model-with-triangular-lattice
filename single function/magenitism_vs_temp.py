'''
这个绘制磁矩和磁化率随温度变化的图像，生成两张图片'magcos_vs_T.png'和'suscept_vs_T.png'
为了运行更快，演化次数已被减少至500+2**12
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
            self.spin_config[self.spin_index[i]]=list(np.random.random(int(N/2))*2*np.pi)[i]#构造xy model的自旋取向random.random取值为0到1
            #self.spin_config[self.spin_index[i]]=list(np.zeros(int(N/2)))[i]
        for i in range(self.num_spins):#让不用来构建自旋的点都为负数
            if i in self.spin_index:
                continue
            else:
                self.spin_config[i]=-2
        self.temperature = temperature
        self.energy = np.sum(self.get_energy())/(2*self.width*self.length)#求单个能量
        self.magcos = []
        self.totalmag=[]
        self.Cv = []
        self.magcos_suscept=[]#磁化率
        self.totalmag_suscept=[]
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
        dic_thermal_t['magcos']=[]
        dic_thermal_t['magsin']=[]
        dic_thermal_t['mag**2']=[]
        dic_thermal_t['totalmag']=[]
        #dic_thermal_t['vortex_density']=[]
        beta = 1.0/self.temperature
        for k in range(max1_nsweeps):
            self.sweep()     
     #       energy = np.sum(self.get_energy())/(2*self.width*self.length)
            #dic_thermal_t['energy'] += [energy]
            #dic_thermal_t['magnetism']+=[self.get_magcos()/(self.width*self.length)]
            #dic_thermal_t['vortex_density'] += [np.sum(self.vortex_density())/(self.width*self.length)]
            if (k>1000)  or k == max1_nsweeps-1:
    #        if ( ( abs(energy)==0 or abs(energy-energy_temp)/abs(energy)<1e-4 ) & (k>9000) ) or k == max1_nsweeps-1 :
                print('\nequilibrium state is reached at T=%.3f'%self.temperature)
                print('#sweep=%i'%k)
    #            #print('energy=%.3f'%energy)
                if show:#show用来控制是否需要将达到平衡后的状态画图输出，默认false
                    self.show()
                break
    #        energy_temp = energy
            
        
        
            #绘制能量,磁矩-步数图像
        after_steps=2**12
        for ki in range(int(after_steps)):
            self.sweep()
            dic_thermal_t['magcos']+=[self.get_magcos()]
            dic_thermal_t['magsin']+=[self.get_magsin()]
            dic_thermal_t['mag**2']+=[self.get_magcos()**2+self.get_magsin()**2]
            #dic_thermal_t['totalmag']+=[np.sqrt(self.get_magcos()**2 +self.get_magsin()**2)/self.width/self.length)]
            #dic_thermal_t['vortex_density']+=[np.sum(self.vortex_density())/(self.width*self.length)]
            if show_after  & (ki%1e3 ==0):#equilibrate中的show_after参数控制是否输出每隔1000步的态，默认false
                print('#sweeps=%i'% (ki+1))
   #             print('energy=%.3f'%energy)
                self.show()
        totalmag2=mag2=0
        self.magcos=np.mean(dic_thermal_t['magcos'])
        mag2=np.average(np.power(dic_thermal_t['magcos'],2))
        self.magcos_suscept= (mag2-self.magcos**2)*beta 
        self.totalmag=np.sqrt(np.average(dic_thermal_t['magcos'])**2+np.average(dic_thermal_t['magsin'])**2)/self.width/self.length
        totalmag2=np.average(dic_thermal_t['mag**2'])/(self.width*self.length)**2
        self.totalmag_suscept= (totalmag2-self.totalmag**2)*beta*self.width*self.length
 
        
        

    def annealing(self,T_init=2.5,T_final=0.1,nsteps = 20,show_equi=False):#同一个原始构型，不同温度下演化
        dic_thermal = {}
        dic_thermal['temperature']=np.linspace(T_init,T_final,nsteps)#np.linspace返回nsteps个处于上下限中的均匀分布的值
        dic_thermal['energy']=[]
        dic_thermal['Cv']=[]
        dic_thermal['magcos']=[]
        dic_thermal['totalmag']=[]
        dic_thermal['magcos_suscept']=[]
        dic_thermal['totalmag_suscept']=[]
        for T in dic_thermal['temperature']:
            #self.show()#用于检查每个温度下的初始构型是不是上一个温度的最终演化构型
            #XY_triangular(width=10)
            self.equilibrate(temperature=T,equili_yes=False)
            if show_equi:
                self.show()
            dic_thermal['magcos'] += [self.magcos]
            dic_thermal['magcos_suscept']+= [self.magcos_suscept]
            dic_thermal['totalmag']+=[self.totalmag]
            dic_thermal['totalmag_suscept']+= [self.totalmag_suscept]

            
        

        plt.figure(figsize=(5,4), dpi=200)
        #yene = np.array(dic_thermal['Cv'])
        #yene_smooth = make_interp_spline(x, yene)(x_smooth)#用于平滑函数
        #plt.plot(x_smooth, yene_smooth)
        plt.plot(dic_thermal['temperature'],np.abs(dic_thermal['magcos']))
        plt.scatter(dic_thermal['temperature'],np.abs(dic_thermal['magcos']))
        plt.ylabel(r'$\langle M_x \rangle$')
        plt.xlabel('T')
        plt.savefig('magcos_vs_T.png', bbox_inches='tight')
        plt.show()
        
        
        plt.figure(figsize=(5,4), dpi=200)
        #ycv = np.array(dic_thermal['Cv'])
        #ycv_smooth = make_interp_spline(x, ycv)(x_smooth)#用于平滑函数
        #plt.plot(x_smooth, ycv_smooth)
        plt.plot(dic_thermal['temperature'],dic_thermal['magcos_suscept'])
        plt.scatter(dic_thermal['temperature'],dic_thermal['magcos_suscept'])
        plt.ylabel(r'x-mag susceptibility')
        plt.xlabel('T')
        plt.savefig('suscept_vs_T.png', bbox_inches='tight')
        plt.show()
        
        plt.figure(figsize=(5,4), dpi=200)
        #yene = np.array(dic_thermal['Cv'])
        #yene_smooth = make_interp_spline(x, yene)(x_smooth)#用于平滑函数
        #plt.plot(x_smooth, yene_smooth)
        plt.plot(dic_thermal['temperature'],np.abs(dic_thermal['totalmag']))
        plt.scatter(dic_thermal['temperature'],np.abs(dic_thermal['totalmag']))
        plt.ylabel(r'$\langle M \rangle$')
        plt.xlabel('T')
        plt.savefig('totalmag_vs_T.png', bbox_inches='tight')
        plt.show()
        
        plt.figure(figsize=(5,4), dpi=200)
        #ycv = np.array(dic_thermal['Cv'])
        #ycv_smooth = make_interp_spline(x, ycv)(x_smooth)#用于平滑函数
        #plt.plot(x_smooth, ycv_smooth)
        plt.plot(dic_thermal['temperature'][2:],dic_thermal['totalmag_suscept'][2:])
        plt.scatter(dic_thermal['temperature'][2:],dic_thermal['totalmag_suscept'][2:])
        plt.ylabel(r'totalmag susceptibility')
        plt.xlabel('T')
        plt.savefig('totalsuscept_vs_T.png', bbox_inches='tight')
        plt.show()
        
        
    
        
        
        return dic_thermal
    
        


    def show(self,colored=False):#图形的绘制都是从左下角零点开始，而数据举证表示出来时第一个点在左上角，要翻转对应，无论方格子或箭头
        config_matrix = np.reshape(self.spin_config,(self.length,self.width*2))#self.list2matrix(self.spin_config)
        X, Y = np.meshgrid(np.arange(0, 2 * np.pi, 1/self.width*np.pi), np.arange(0,  np.pi, 1/self.length*np.pi))#np.meshgrid(np.arange(0,self.width*2 ),np.arange(0, self.width*2))
        U=np.zeros((self.length,self.width*2))
        V=np.zeros((self.length,self.width*2))
        x_tri=np.zeros(int(self.width*self.length))#用于只画出三角晶格格点的散点图
        y_tri=np.zeros(int(self.width*self.length))
        xi=0
        for i in range(self.length):
            for j in range(self.width*2):
                if config_matrix[i][j]<0:
                    U[i][j]=0
                    V[i][j]=0
                else:
                    U[i][j]=np.cos(config_matrix[i][j])
                    V[i][j]=np.sin(config_matrix[i][j])
                    x_tri[xi]=j*np.pi*1/self.width
                    y_tri[xi]=i*np.pi*1/self.length
                    xi+=1
                    
        #print(U)
        fig1, ax1 = plt.subplots(figsize=(8,8/1.2),dpi=100)#500
        M = np.hypot(np.cos(X),np.sin(Y))#此参数用于quiver中控制不同区域的颜色
        #fig1.figure(figsize=(4,4), dpi=100)
        arrow_size=10#70
        #plt.title('Arrows scale with plot width, not view')
        Q = ax1.quiver(X, Y, U, V, M,angles="uv",pivot="middle")
        #Q = ax1.quiver(X, Y, U, V, M,angles="uv",units='dots',scale_units="dots",scale=1.0/arrow_size,pivot="tip",linewidth=3,width=3,headwidth=arrow_size*0.3,headlength=0.6*arrow_size,headaxislength=arrow_size*0.4,zorder=100)
        #plt.colorbar()
        qk = ax1.quiverkey(Q, 0.5, 0.1, 1, r'$spin$', labelpos='E', coordinates='figure')
        ax1.scatter(x_tri, y_tri, color='0', s=1)#画晶格散点图
        ax1.set_title('T=%.2f'%self.temperature+', #spins='+str(self.width)+'x'+str(self.length))
        ax1.axis('off')
        plt.savefig('f{:}_1.png'.format(self.temperature), bbox_inches='tight')
        plt.show()

        #方格子
        plt.figure(figsize=(4,4), dpi=100)
        x_squre=np.zeros((self.length+1,self.width+1))#方格子方案1，只取有自旋的点拿出来从新排列
        y_squre=np.zeros((self.length+1,self.width+1))
        for i in range(self.length+1):
            for j in range(self.width+1):
                x_squre[i][j]=j
                y_squre[i][j]=i
        c_matr=[]
        spin_middle_value=0
        for i in self.spin_index:
            spin_middle_value=self.spin_config[i]
            if spin_middle_value > 2*np.pi:
                spin_middle_value -= 2*np.pi
            c_matr.append(spin_middle_value)
        c_value_cos=np.cos(np.reshape(c_matr,(self.length,self.width)))#使用cos(x/2)倒是能保证整个区间内为单调函数，但是太不直观了
        #print(c_value)
        #plt.plot(x_squre.ravel(),y_squre.ravel(),"ko")这个显示图中标出矩形点
        cs=plt.pcolormesh(x_squre, y_squre,c_value_cos,cmap ="rainbow",vmin = -1, vmax = +1,rasterized=True,edgecolors='face',alpha=0.6)#"flag" or "rainbow" ,可设置最大最小值  并在后面加上语句  plt.colorbar(cs)显示色柱
        plt.title('T=%.2f'%self.temperature+',cos, #spins='+str(self.width)+'x'+str(self.length))
        plt.colorbar(cs)
        plt.axis('off')      
        plt.savefig('f{:}_2.png'.format(self.temperature), bbox_inches='tight')
        plt.show()
        
        plt.figure(figsize=(4,4), dpi=100)
        c_value_sin=np.sin(np.reshape(c_matr,(self.length,self.width)))
        #print(c_value)
        #plt.plot(x_squre.ravel(),y_squre.ravel(),"ko")这个显示图中标出矩形点
        cs=plt.pcolormesh(x_squre, y_squre,c_value_sin,cmap ="rainbow",vmin = -1, vmax = +1,rasterized=True,edgecolors='face',alpha=0.6)#"flag" or "rainbow" ,可设置最大最小值  并在后面加上语句  plt.colorbar(cs)显示色柱
        plt.title('T=%.2f'%self.temperature+',sin, #spins='+str(self.width)+'x'+str(self.length))
        plt.colorbar(cs)
        plt.axis('off')
        plt.savefig('f{:}_3.png'.format(self.temperature), bbox_inches='tight')
        



Tstep=30
Ti=0.02
Tf=3.0
dic_co={}
a = XY_triangular(width=10)
a.annealing(T_init=Ti,T_final=Tf,nsteps = Tstep,show_equi=False)
