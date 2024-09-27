import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.cm import get_cmap
from matplotlib import animation

SIZE = 100
LOSS = 0.1
LOSSL = 90
ez=np.zeros(SIZE)
hy=np.zeros(SIZE)
ceze=np.zeros(SIZE)
cezh=np.zeros(SIZE)
chyh=np.zeros(SIZE-1)
chye=np.zeros(SIZE-1)
epsr=4.0
imp0=377.0
maxTime=550
time=0
data=[]
EzP=[]


Cdtds = 1.0 / math.sqrt(2)
width=1.0
delay=1.0
tfsfBoundary = 20
ppw=20

def Plot1d(arr: np.ndarray, qTime: int, save_dir: str = "C:/FTDT/1dPlot"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(10, 5))
    plt.plot(arr, label=f'Last values at Time {qTime}')
    plt.xlabel('Index')
    plt.ylabel('Field Value')
    plt.title(f'Last Values of Ez at Time {qTime}')
    plt.legend()

    save_path = os.path.join(save_dir, f"1dPlot{qTime}.png")
    plt.savefig(save_path)
    plt.close()

def ezINC(qTime,location):
    arg = math.pi * ((Cdtds * qTime - location) / ppw - 1.0)
    arg2 = arg * arg
    return (1.0 - 2.0 * arg2) * math.exp(-arg2)
    
for mm in range(SIZE):
    ez[mm] = 0

for mm in range(SIZE-1):
    hy[mm] = 0

for mm in range(SIZE):
        ceze[mm] = 1.0
        cezh[mm] = imp0
for mm in range(SIZE-1):
        chyh[mm] = 1.0
        chye[mm] = 1.0 / imp0





for qTime in range(maxTime):

    for mm in range(SIZE - 1):
        hy[mm] = chyh[mm] * hy[mm] + chye[mm] * (ez[mm+1]-ez[mm])
    
    hy[tfsfBoundary-1] -= ezINC(qTime, 0) * chye[tfsfBoundary-1]
    

    tempL1 = math.sqrt((cezh[0] * chye[0]))
    tempL2 = 1 / tempL1 + 2 + tempL1 
    abcCleft = np.zeros(3)
    abcCleft[0] = -(1 / tempL1 - 2 + tempL1) / tempL2
    abcCleft[1] = -2 * (tempL1 - 1 / tempL1) / tempL2
    abcCleft[2] = 4 * (tempL1 + 1 / tempL1) / tempL2

    tempR1 = math.sqrt(cezh[SIZE-1] * chye[SIZE-2])
    tempR2 = 1 / tempR1 + 2 + tempR1
    abcCright = np.zeros(3)
    abcCright[0] = -(1 / tempR1 - 2 + tempR1) / tempR2
    abcCright[1] = -2 * (tempR1 - 1 / tempR1) / tempR2
    abcCright[2] = 4 * (tempR1 + 1 / tempR1) / tempR2

    ezOldleft1 = np.zeros(3)
    ezOldleft2 = np.zeros(3)
    ezOldright1 = np.zeros(3)
    ezOldright2 = np.zeros(3)

    ez[0] = abcCleft[0] * (ez[2] + ezOldleft2[0]) + abcCleft[1] * (ezOldleft1[0] + ezOldleft1[2] - ez[1] - ezOldleft2[1] ) + abcCleft[2] * ezOldleft1[1] - ezOldleft2[2]
    ez[SIZE-1] = abcCright[0] * (ez[SIZE-3] + ezOldright2[0]) + abcCright[1] * (ezOldright1[0] + ezOldright1[2] - ez[SIZE-2] - ezOldright2[1]) + abcCright[2] * ezOldright1[1] - ezOldright2[2]

    for mm in range(0,2):
        ezOldleft2[mm] = ezOldleft1[mm]
        ezOldleft1[mm] = ez[mm]

        ezOldright2[mm] = ezOldright1[mm]
        ezOldright1[mm] = ez[SIZE-1-mm]

    for mm in range(1,SIZE):
        ez[mm] = ceze[mm] * ez[mm] + cezh[mm] * (hy[mm]-hy[mm-1])

   

    ez[tfsfBoundary] += ezINC(qTime +0.5, -0.5)

    Plot1d(ez,qTime)

    if qTime % 50 == 0:
        print(qTime)



