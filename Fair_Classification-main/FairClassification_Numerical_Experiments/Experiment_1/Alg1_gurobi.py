import sys
import math
import random
import itertools
from gurobipy import *
import os
import xlrd
import csv
import random
import numpy as np
import scipy as sp
import pandas as pd
import time

title="wine_5000"
book = xlrd.open_workbook(title+".xlsx")
sh= book.sheet_by_name(title)
global df
df = pd.DataFrame(columns=('rho','time','obj'))
np.random.seed(1)

y =[]
X =[]
i = 0
while True:
    try:
        y.append(sh.cell_value(i, 0))            
        j=0
        rows=[0]*2
        while True:
            try:             
                rows[int(sh.cell_value(i, 2*j+1))-1]=sh.cell_value(i, 2*j+2)  
                j=j+1
            except IndexError:
                break
            except ValueError:
                break
        X.append(rows)          
        i = i + 1
    except IndexError:
        break

I=2
N=len(y)
mm=len(X[0][:])
M=mm*mm*I

start = time.time()

def Model1(wvals,uvals,bval,rhopara):

    mb = Model()
    bC=1e2

    D1=0.0
    D2=0.0
    N1=[]
    N2=[]

    for j in range(N):
        if X[j][0]==1:
            D1=D1+1.0
            N1.append(j)
        else:        
            D2=D2+1.0
            N2.append(j)

    eabs = mb.addVar( lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='eabs')
    eabs1 = mb.addVar( lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='eabs1')
    zvars = mb.addVars(N,lb=0.0, ub=1.0, vtype=GRB.BINARY, name='Z')

    mb.setObjective(rhopara*eabs+quicksum(1.0/float(N)*(uvals[j]-1)*zvars[j] for j in range(N)), GRB.MINIMIZE)
    
    mb.update()

    mb.addConstr(eabs1 == quicksum(zvars[j] for j in N1)/D1 - quicksum(zvars[j] for j in N2)/D2)
    mb.addGenConstrAbs(eabs, eabs1, 'absconstr')

    mb.update()

    mb.params.OutputFlag = 0
    mb.params.timelimit = 600 
    mb.optimize()
    zvals= mb.getAttr('x', zvars)
    evals = eabs.x
    evals1 = eabs1.x
    
    return mb.objVal,zvals,evals,evals1
    

iter1=0
for rhopara in [0.01,0.03,0.05,0.1,0.2,0.5,0.8,1.0,2.0,3.0,5.0,10.0]:    

    uvals=[1.0]*N
    wvals=[1.0]*N
      
    bval=100
    r=10.0
    for j in range(N):
        uvals[j]=np.random.uniform(0,r)
        wvals[j]=np.random.uniform(-r,r)

    bestobj,zvals,evals,evals1=Model1(wvals,uvals,bval,rhopara)


    totaltime=time.time()-start

    df.loc[iter1] =np.array([rhopara,totaltime,bestobj])
    df.to_csv(title+"_gur.csv")
    
    iter1=iter1+1
        


