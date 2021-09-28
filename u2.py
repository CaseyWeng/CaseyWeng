from pykrige.ok3d import OrdinaryKriging3D
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from pykrige.kriging_tools import write_asc_grid #Spatial_interpolation-CyberGISX
import pykrige.kriging_tools as kt #Spatial_interpolation-CyberGISX

carpeta = "D:\python training\m.csv"
df = pd.read_csv(carpeta, sep=",")
df.head(1)
print("No of datas:",df['u2 (MPa)'].shape)
print("xmin:",df['E'].min(),"xmax:",df['E'].max(),"ymin:",df['N'].min(),"ymax:",df['N'].max(),"zmin:",df['Z(m)'].min(),"zmax:",df['Z(m)'].max())
print(round(df['u2 (MPa)'].var(),2))

# **conversion of vairables to ndarray**
x = np.array(df['E'])
y = np.array(df['N'])
z = np.array(df['Z(m)'])
val = np.array(df['u2 (MPa)'])

# **variograma**
ok3d = OrdinaryKriging3D(x,y,z,val,variogram_model='gaussian',nlags=10,enable_plotting=True,verbose=True)

# **grid set**
gridx = np.linspace(171160,174743,num=35,endpoint=False)
gridy = np.linspace(2650705,2656410,num=57,endpoint=False)
gridz = np.linspace(0.3,3,num=10,endpoint=False)
zg,yg,xg = np.meshgrid(gridz,gridy,gridx,indexing='ij')
fig = plt.figure(figsize=(15,15))
plot1 = fig.add_subplot(111)
a = plt.scatter(xg,yg)
plt.xlabel("E")
plt.ylabel("N")
plt.title("Grid X-Y")
plt.grid(True)

fig3d = plt.figure(figsize=(15,15))
plot3d = fig3d.add_subplot(111,projection='3d')
plot3d.scatter(xg,yg,zg)
plot3d.set_xlabel("E")
plot3d.set_ylabel("N")
plot3d.set_zlabel("Z")
plt.show()

k3d1,ss3d = ok3d.execute('grid',gridx,gridy,gridz)# ok3d1是结果，给出了每个网格点处对应的值

a = np.round(k3d1,2)#输出的结果
b = np.round(gridx,2)
c = np.round(gridy,2)
d = np.round(gridz,2)

#pykrige.kriging_tools.write_asc_grid(gridx,gridy,gridz,k3d1)
[z1,y1,x1,qc] = np.meshgrid(a,b,c,d)
print([x1,y1,z1,qc])





