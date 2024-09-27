import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import os


SIZEX = 100
SIZEY = 80
FX = 5
FY = 5
LX = 95
LY = 75
maxTime = 1000
cdtds = 1 / math.sqrt(2)
imp0 = 377
ppw = 30 #points per wavelength

ez1d=np.zeros(SIZEX)
hy1d=np.zeros(SIZEX)
ceze1d=np.zeros(SIZEX)
cezh1d=np.zeros(SIZEX)
chyh1d=np.zeros(SIZEX)
chye1d=np.zeros(SIZEX)
ceye1d=np.zeros(SIZEX)
ceyh1d=np.zeros(SIZEX)
chzh1d=np.zeros(SIZEX)
chze1d=np.zeros(SIZEX)

ezOldLeft = np.zeros(SIZEY)
ezOldRight = np.zeros(SIZEY)
ezOldTop = np.zeros(SIZEX)
ezOldBottom = np.zeros(SIZEX)

ezx = np.zeros((SIZEY,SIZEX))
ezy = np.zeros((SIZEY,SIZEX))

ez = np.zeros((SIZEY,SIZEX))

hx = np.zeros((SIZEY,SIZEX))
chxh = np.zeros((SIZEY,SIZEX))
chxe = np.zeros((SIZEY,SIZEX))

hy = np.zeros((SIZEY,SIZEX))
chyh = np.zeros((SIZEY,SIZEX))
chye = np.zeros((SIZEY,SIZEX))

epsilon0 = (1/(36*math.pi)) * 0.000000001
mu0 = 4*math.pi* 0.0000001
epsilon = np.ones((SIZEY,SIZEX)) * epsilon0
mu = np.ones((SIZEY,SIZEX)) * mu0

sigmax = np.zeros((SIZEY,SIZEX))
sigmay = np.zeros((SIZEY,SIZEX))

boundW = 5
gradingorder = 6
reflectcoeff = 0.000001
c = 300000000

SS = 1*10**(-6) #spacial step
TS = cdtds * (SS/c) #temporal step

sigmamax = abs((math.log10(reflectcoeff) * (gradingorder+1) * epsilon0 * c) / (2 * boundW * SS))
boundfact1 = ((epsilon[boundW, SIZEX//2] / epsilon0) * sigmamax) / ((boundW**gradingorder) * (gradingorder+1))
boundfact2 = ((epsilon[SIZEY-boundW, SIZEX//2] / epsilon0) * sigmamax) / ((boundW**gradingorder) * (gradingorder+1))
boundfact3 = ((epsilon[SIZEY//2, boundW] / epsilon0) * sigmamax) / ((boundW**gradingorder) * (gradingorder+1))
boundfact4 = ((epsilon[SIZEY//2, SIZEX-boundW] / epsilon0) * sigmamax) / ((boundW**gradingorder) * (gradingorder+1))
temp = np.arange(0, boundW + 1, 1)

for mm in range(1,SIZEX,1):
    sigmay[boundW::-1, mm ] = boundfact1 * ((temp + 0.5)**(gradingorder + 1) - (temp - 0.5)**(gradingorder + 1))
    sigmay[ SIZEY-boundW-1:SIZEY:1,mm] = boundfact2 * ((temp + 0.5)**(gradingorder + 1) - (temp - 0.5)**(gradingorder + 1))
for nn in range(1,SIZEY,1):
    sigmax[nn, boundW::-1] = boundfact3 * ((temp + 0.5)**(gradingorder + 1) - (temp - 0.5)**(gradingorder + 1)) 
    sigmax[nn, SIZEX-boundW-1:SIZEX:1] = boundfact4 * ((temp + 0.5)**(gradingorder + 1) - (temp - 0.5)**(gradingorder + 1))

sigma_stary = (sigmay * mu) / epsilon
sigma_starx = (sigmax * mu) / epsilon

chyh = ((mu - 0.5 * (TS) * sigma_stary) / (mu + 0.5 * (TS) * sigma_stary))
chye = (TS/SS) / (mu + 0.5 * TS * sigma_stary)
chxh = ((mu - 0.5 * TS * sigma_starx) / (mu + 0.5 * TS * sigma_starx))
chxe = (TS/SS) / (mu + 0.5 * TS * sigma_starx)
#TS/SS = 2.35x10^-9, chye1=1.48*10^-11
ceye = ((epsilon - 0.5 * TS * sigmay) / (epsilon + 0.5 * TS * sigmay))
ceyh = (TS/SS) / (epsilon + 0.5 * TS * sigmay)
cexe = ((epsilon - 0.5 * TS * sigmax) / (epsilon + 0.5 * TS * sigmax))
cexh = (TS/SS) / (epsilon + 0.5 * TS * sigmax)

#chyh1d = ((mu[:,0] - 0.5 * (TS) * sigma_stary[0,:]) / (mu[:,0] + 0.5 * (TS) * sigma_stary[:,0]))
#chye1d = (TS/SS) / (mu[:,0] + 0.5 * TS * sigma_stary[:,0])
#ceze1d = ((epsilon[0,:] - 0.5 * TS * sigmax[0,:]) / (epsilon[0,:] + 0.5 * TS * sigmax[0,:]))
#cezh1d = (TS/SS) / (epsilon[0,:] + 0.5 * TS * sigmax[0,:])


def ezINC(qTime,location):
    arg = 1 * math.pi * ((cdtds * qTime - location) / ppw - 1.0  ) 
    arg2 = arg * arg
    return (1.0 - 2.0 * arg2) * math.exp(-arg2)

def ezINC2(qTime, location, wavelength):
    k = 2 * math.pi / wavelength  
    omega = 2 * math.pi / (wavelength * cdtds)  
    return math.sin(k * location - omega * qTime)


class Plots():

    def __init__(self, arr: np.ndarray, qTime: int, save_dir: str):
        self.arr = arr
        self.qTime = qTime
        self.save_dir = save_dir

    def Plot1d(self):

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        plt.figure(figsize=(10, 5))
        plt.plot(self.arr, label=f'Last values at Time {self.qTime}')
        plt.xlabel('Index')
        plt.ylabel('Field Value')
        plt.title(f'Last Values of Ez at Time {self.qTime}')
        plt.legend()

        save_path = os.path.join(self.save_dir, f"1dPlot{self.qTime}.png")
        plt.savefig(save_path)
        plt.close()

        if qTime % 50 == 0:
            print(qTime)

    def heatmap(self):
   
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        plt.figure(figsize=(8,6))
        plt.imshow(self.arr, cmap='viridis', origin='lower',vmax=2,vmin=-2)  
        plt.colorbar()
        plt.title(f"Heatmap at Time {self.qTime}")
        save_path = os.path.join(self.save_dir, f"heatmap2_{self.qTime}.png")
        plt.savefig(save_path)
        plt.close()

        if qTime % 50 == 0:
            print(qTime)
        
    def Plot3d(self):
 
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        x = np.arange(self.arr.shape[0])
        y = np.arange(self.arr.shape[1])
        X, Y = np.meshgrid(x, y)
        Z = self.arr.T 

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(Y, X, Z, cmap='viridis', edgecolor='none')
        ax.set_xlim(0, self.arr.shape[1])
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('E_z')
        ax.set_title(f'3D Surface Plot at Time {self.qTime}')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        if qTime % 50 == 0:
            print(qTime)

        save_path = os.path.join(self.save_dir, f"3d_surface_{self.qTime}.png")
        plt.savefig(save_path)
        plt.close()
    
    def slitPlot(self):

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
        lasty = self.arr[:, 100:].mean(axis=1) 
 
        plt.figure(figsize=(10, 5))
        plt.plot(lasty, range(lasty.shape[0]), label=f'Last values at Time {self.qTime}')
        plt.xlabel('Index')
        plt.ylabel('Field Value')
        plt.title(f'Last Values of Ez at Time {self.qTime}')
        plt.legend()
        
        save_path = os.path.join(self.save_dir, f"slitPlot{self.qTime}.png")
        plt.savefig(save_path)
        plt.close()

class PECs():
    
    def __init__(self,radius,x,y,x2=None): #note x2 is only needed for Double Slit function
        self.radius = radius
        self.x = x
        self.x2 =x2
        self.y = y

    def PEC_Circle(self):

        radius2 = self.radius * self.radius
        for mm in range(1, SIZEX - 1):
            xLocation = mm - self.x
            for nn in range(1, SIZEY - 1):
                yLocation = nn - self.y
                if xLocation * xLocation + yLocation * yLocation < radius2:
                    cexe[nn,mm] = 0
                    cexh[nn,mm] = 0
                    ceye[nn,mm] = 0 
                    ceyh[nn,mm] = 0
                    chxe[nn,mm] = 0
                    chxh[nn,mm] = 0
                    chye[nn, mm] = 0
                    chyh[nn, mm] = 0
    
    def Dia_Circle(self):

        radius2 = self.radius * self.radius
        for mm in range(1, SIZEX - 1):
            xLocation = mm - self.x
            for nn in range(1, SIZEY - 1):
                yLocation = nn - self.y
                if xLocation * xLocation + yLocation * yLocation < radius2:
                    cexe[nn,mm] = 1 
                    cexh[nn,mm] = cdtds * imp0 / math.sqrt(100)
                    ceye[nn,mm] = 1 
                    ceyh[nn,mm] = cdtds * imp0 / math.sqrt(100)
                    chxe[nn,mm] = cdtds / (imp0)
                    chxh[nn,mm] = 1
                    chye[nn, mm] = cdtds / (imp0 )
                    chyh[nn, mm] = 1

    def PEC_Rec(self,width,height):

        for mm in range(1,SIZEX - 1):
            xlocation = mm - self.x
            for nn in range(1, SIZEY - 1):
                ylocation = nn - self.y
                if abs((xlocation/width) + (ylocation/height)) + abs((xlocation/width) - (ylocation/height)) < self.radius:
                    chxe[nn,mm] = 0
                    chxh[nn,mm] = 0
                    chye[nn, mm] = 0
                    chyh[nn, mm] = 0
                    cexe[nn, mm] = 0
                    cexh[nn, mm] = 0
                    ceye[nn,mm] = 0
                    ceyh[nn,mm] = 0

    def Dia_Rec(self,width,height):

        for mm in range(1,SIZEX - 1):
            xlocation = mm - self.x
            for nn in range(1, SIZEY - 1):
                ylocation = nn - self.y
                if abs((xlocation/width) + (ylocation/height)) + abs((xlocation/width) - (ylocation/height)) < self.radius:
                    chxe[nn,mm] = cdtds / imp0
                    chxh[nn,mm] = 1
                    chye[nn, mm] = cdtds / imp0
                    chyh[nn, mm] = 1
                    cexe[nn, mm] = 1
                    cexh[nn, mm] = cdtds * imp0 / math.sqrt(10)
                    ceye[nn,mm] = 1
                    ceyh[nn,mm] = cdtds * imp0 / math.sqrt(10)

    def PEC_Line(self, x_start, y_start, x_end, y_end):
        # Check if the line is vertical or horizontal
        if x_start == x_end:  # Vertical line
            for y in range(min(y_start, y_end), max(y_start, y_end) + 1):
                chxe[y, x_start] = cdtds / imp0 *0
                chxh[y, x_start] = 1 *0
                chye[y, x_start] = cdtds / imp0 *0
                chyh[y, x_start] = 1 *0

                cexe[y, x_start] = 1 *0
                cexh[y, x_start] = cdtds * imp0 / math.sqrt(10) *0
                ceye[y, x_start] = 1 *0
                ceyh[y, x_start] = cdtds * imp0 / math.sqrt(10) *0

        elif y_start == y_end:  # Horizontal line
            for x in range(min(x_start, x_end), max(x_start, x_end) + 1):
                chxe[y_start, x] = cdtds / imp0
                chxh[y_start, x] = 1
                chye[y_start, x] = cdtds / imp0
                chyh[y_start, x] = 1
                
                cexe[y_start, x] = 1
                cexh[y_start, x] = cdtds * imp0 / math.sqrt(10)
                ceye[y_start, x] = 1
                ceyh[y_start, x] = cdtds * imp0 / math.sqrt(10)

    def Lens_Like_PEC(self, EPSR):
        r2 = self.radius * self.radius
        
        for mm in range(SIZEX):
            x12 = mm - self.x
            x12 = x12 * x12
            x22 = mm - self.x2
            x22 = x22 * x22
            
            for nn in range(SIZEY):
                y2 = nn - self.y
                y2 = y2 * y2
                
                if x12 + y2 < r2 and x22 + y2 < r2:
                    
                    cexe[nn, mm] = 1.0
                    cexh[nn, mm] = cdtds * imp0 / math.sqrt(EPSR)
                    ceye[nn,mm] = 1.0
                    ceyh[nn,mm] = cdtds * imp0 / math.sqrt(EPSR)
                else:
                    cexe[nn, mm] = 1.0
                    cexh[nn, mm] = cdtds * imp0
                    ceye[nn,mm] = 1.0
                    ceyh[nn,mm] = cdtds * imp0

        
        for mm in range(SIZEX):
            for nn in range(SIZEY - 1):
                chxh[nn, mm] = 1.0
                chxe[nn, mm] = cdtds / imp0

        for mm in range(SIZEX - 1):
            for nn in range(SIZEY):
                chyh[nn, mm] = 1.0
                chye[nn, mm] = cdtds / imp0

    def PEC45(self, x_start, y_start, x_end, y_end):

        # Check if the points form a 45-degree line
        if abs(x_end - x_start) != abs(y_end - y_start):
            raise ValueError("The provided points do not form a 45-degree line.")
        
        # Iterate over the range from start to end
        x = x_start
        y = y_start

        x_direction = 1 if x_end > x_start else -1  # Determine the direction (positive or negative)
        y_direction = 1 if y_end > y_start else -1  # for both x and y

        while x != x_end and y != y_end:
            # Set the field components to zero to represent PEC at the current (y, x) location
            chxe[y, x] = 0
            chxh[y, x] = 0
            chxe[y, x + 1] = 0
            chxh[y, x + 1] = 0
            chye[y + 1, x] = 0
            chyh[y + 1, x] = 0
            chye[y, x] = 0
            chyh[y, x] = 0
            cexe[y, x] = 0
            cexh[y, x] = 0
            ceye[y, x] = 0
            ceyh[y, x] = 0
            
            # Move diagonally in both x and y directions (for 45-degree line)
            x += x_direction
            y += y_direction

class FieldUpdates():

    def __init__(self,SIZEX,SIZEY):
        self.SIZEX = SIZEX
        self.SIZEY = SIZEY

    def UpdateMag(self):

        for mm in range(SIZEX-1):
             for nn in range(SIZEY-1):                             
                    hy[nn,mm] = chyh[nn,mm] * hy[nn,mm] + chye[nn,mm] * (ezx[nn+1,mm] - ezx[nn,mm] + ezy[nn+1,mm] - ezy[nn,mm])
                    hx[nn,mm] = chxh[nn,mm] * hx[nn,mm] - chxe[nn,mm] * (ezx[nn,mm+1] - ezx[nn,mm] + ezy[nn,mm+1] - ezy[nn,mm])
    
    def Update1dMag(self):

        for mm in range(SIZEX-1):
            hy1d[mm] = chyh1d[mm] * hy1d[mm] + chye1d[mm] * (ez1d[mm+1]-ez1d[mm])

    def Update1dElc(self):

        for mm in range(1,SIZEX):
            ez1d[mm] = ceze1d[mm] * ez1d[mm] + cezh1d[mm] * (hy1d[mm]-hy1d[mm-1])
        
    def UpdateElc(self):
        
        for mm in range(SIZEX):
             for nn in range(SIZEY):
                    ezx[nn,mm] = cexe[nn,mm] * ezx[nn,mm] + cexh[nn,mm] * (hx[nn,mm-1] - hx[nn,mm])
                    ezy[nn,mm] = ceye[nn,mm] * ezy[nn,mm] + ceyh[nn,mm] * (hy[nn,mm] - hy[nn-1,mm])

for mm in range(SIZEX):
                ceze1d[mm] = 1.0
                cezh1d[mm] = cdtds * imp0
                chyh1d[mm] = 1.0
                chye1d[mm] = cdtds / imp0 

class TFSF():

    def CorrectMag(self):

        for mm in range(FX,LX):
            hy[FY,mm] -= 0.001875 * ez1d[mm] 

        for mm in range(FX,LX):
            hy[LY-1,mm] += 0.001875 * ez1d[mm] 

        for nn in range(FY,LY):
            hx[nn,FX-1] += 0.001875 * ez1d[FX-1] 

        for nn in range(FY,LY):
            hx[nn,LX-1] -= 0.001875 * ez1d[LX-1]

    def CorrectElc(self):

        for nn in range(FY,LY):
            ezy[nn, FX] -= 266.58 * hy1d[FX]  
            ezy[nn, LX] += 266.58 * hy1d[LX]  
        
            ezx[nn, FX] -= 266.58 * hy1d[FX]  
            ezx[nn, LX] += 266.58 * hy1d[LX]
                   
for qTime in range(maxTime):

    PEC1 = PECs(1,20,20)
    PEC1.PEC_Line(20,20,20,60)
    PEC1.PEC_Line(30,20,30,60)
    PEC1.PEC_Line(40,20,40,60)
    PEC1.PEC_Line(50,20,50,60)
    PEC1.PEC_Line(60,20,60,60)
    PEC1.PEC_Line(70,20,70,60)
    PEC1.PEC_Line(80,20,80,60)
    

    """#Harmonic Resonator?
    PEC2 = PECs(1,50,15)
    PEC2.Dia_Rec(200,10)
    PEC3 = PECs(25,40,45)
    PEC3.Dia_Circle()
    PEC4 = PECs(13,40,45)
    PEC4.PEC_Circle()
    PEC3 = PECs(25,140,45)
    PEC3.Dia_Circle()
    PEC4 = PECs(13,140,45)
    PEC4.PEC_Circle()
    PEC5 = PECs(1,50,85)
    PEC5.Dia_Rec(200,10)
    """

    """ #Waveguide Example, SIZEX = 101, SIZEY = 200, No TFSF, Two Sources 3 units apart one += another -=
    PEC1 = PECs(1,25,110)
    PEC1.PEC_Rec(50,1)
    PEC2 = PECs(1,25,90)
    PEC2.PEC_Rec(50,1)
    PEC3 = PECs(1,75,100)
    PEC3.PEC_Rec(1,200)
    PEC4 = PECs(1,50,156)
    PEC4.PEC_Rec(1,90)
    PEC4 = PECs(1,50,44)
    PEC4.PEC_Rec(1,90)
    """

    """
    PEC1 = PECs(1,140,40) #SIZEX = 200, SIZEY = 200, No TFSF, One Guassian Source With Repeated Waves (0,100,20) PEC Circle Changed To Dielectric = 4
    PEC1.PEC_Rec(110,80)
    PEC2 = PECs(1,110,40)
    PEC2.PEC45(25,120,65,160)
    PEC3 = PECs(25,150,150)
    PEC3.PEC_Circle()
    """

    UpdateFields = FieldUpdates(SIZEY,SIZEX)
    UpdateFields.UpdateMag()
    
    TFSFs = TFSF()
    TFSFs.CorrectMag()       
    UpdateFields.Update1dMag()
    hy1d[FX-1] -= ezINC(qTime, 0) * chye1d[FX-1]
    UpdateFields.Update1dElc()
    ez1d[FX] -= ezINC(qTime -0.5, -0.5) 
    TFSFs.CorrectElc()
    
    UpdateFields.UpdateElc()
   
    #for mm in range(0,1000,20):
        #ezx[20,50] += ezINC(qTime,mm,60)
        #ezy[20,50] += ezINC(qTime,mm,60)
        #ezx[50,35] -= ezINC(qTime,mm,60)
        #ezy[50,35] -= ezINC(qTime,mm,60)
    
    ez = ezx + ezy
    Plot = Plots(ez,qTime,"C:/FTDT/heatmaps2")
    Plot.heatmap()
    
    
    