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


def ModelSVM(lampara):
    mb = Model()
    bC=1e2

    wvars = mb.addVars(mm, lb=-bC, ub=bC, vtype=GRB.CONTINUOUS, name='w')
    uvars = mb.addVars(N, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='u')
    bvar = mb.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='b')
    
    mb.setObjective(quicksum(uvars[j] for j in range(N))+quicksum(lampara*wvars[i]*wvars[i] for i in range(mm)), GRB.MINIMIZE)    
    mb.update()

    for j in range(N):     
        ind=y[j]
        mb.addConstr(quicksum(ind*wvars[i]*X[j][i] for i in range(mm))+ind*bvar>=1.0-uvars[j], 'w b M'+str(j)+str(ind))
    mb.update()

    mb.params.OutputFlag = 0
    mb.params.timelimit = 300
    
    mb.optimize()
    wvals= mb.getAttr('x', wvars)
    bval=bvar.x
    uvals= mb.getAttr('x', uvars)
    return mb.objVal,uvals,wvals,bval

global df


title="synthetic_data"
book = xlrd.open_workbook(title+".xlsx")
sh= book.sheet_by_name(title)

y =[]
X3 =[]
i = 0
while True:
    try:
        y.append(sh.cell_value(i, 0))            
        j=0
        rows=[0]*3
        while True:
            try:
                rows[int(sh.cell_value(i, 2*j+1))-1]=sh.cell_value(i, 2*j+2)  
                j=j+1
            except IndexError:
                break
            except ValueError:
                break
        X3.append(rows)          
        i = i + 1
    except IndexError:
        break
    

X=[]
X0=[]
for i in range(len(X3)):
    x2=[X3[i][1]]+[X3[i][2]]
    X.append(x2)
    X0.append(X3[i][0])


I=2
N=len(y)
mm=len(X[0][:])

C=10
kappa=10
K=0
delta=0.1
a=1
i=0
M=mm*mm*I
    
np.random.seed(1)


for lampara in [0.5]:
    
    bestobj,uvals,wvals,bval=ModelSVM(lampara)

    bestw=wvals
    bestb=bval
    
    count=0

    pred=[0]*N
    zpred=[0.0]*N      
    for j in range(N):
        arr = (sum(bestw[i]*float(X[j][i]) for i in range(mm))+bestb)
        ind= np.sign(arr)
        pred[j]=ind
        if (y[j]==ind):
            count=count+1
            zpred[j]=1.0
                
    expr1=0.0
    expr2=0.0
    D1=0.0
    D2=0.0

    for j in range(N):
        if X0[j]==1:
            D1=D1+1.0
            expr1=expr1+zpred[j]
            
        else:
            D2=D2+1.0
            expr2=expr2+zpred[j]
           
    rate=float(count)*(1.0/float(N))
    Fairness=math.fabs(expr1/D1-expr2/D2)
    
    print("Accuracy:", rate)    
    print("Fairness:", Fairness)
    
    
    

   
