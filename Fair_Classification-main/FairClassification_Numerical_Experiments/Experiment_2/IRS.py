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

def Model1(zvals,rhopara,lampara):
    mb = Model()
    bC=1e2

    wvars = mb.addVars(mm, lb=-bC, ub=bC, vtype=GRB.CONTINUOUS, name='w')
    uvars = mb.addVars(N, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='u')
    bvar = mb.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='b')
    
    mb.setObjective(1.0/float(N)*quicksum((uvars[j]-1.0)*(zvals[j]) for j in range(N))+quicksum(lampara*wvars[i]*wvars[i] for i in range(mm)), GRB.MINIMIZE)
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

    D1=0.0
    D2=0.0
    expr1=0.0
    expr2=0.0
    for j in range(N):

        if X[j][0]==1:
            D1=D1+1.0
            expr1=expr1+zvals[j]
        else:        
            D2=D2+1.0
            expr2=expr2+zvals[j]
        
    return mb.objVal+rhopara*math.fabs(expr1/D1-expr2/D2),uvals,wvals,bval


def Model21(wvals,uvals,rhopara,lampara):
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
            
    zvars = mb.addVars(N,lb=0.0, ub=1.0, vtype=GRB.BINARY, name='Z')

    mb.setObjective(quicksum((1.0/float(N))*(uvals[j]-1.0)*(zvars[j]) for j in range(N))+quicksum(lampara*wvals[i]*wvals[i] for i in range(mm))
                    +quicksum(rhopara*zvars[j]/D1 for j in N1)-quicksum(rhopara*zvars[j]/D2 for j in N2), GRB.MINIMIZE)
            
    mb.update()

    mb.addConstr(quicksum(zvars[j]*D2 for j in N1)-quicksum(zvars[j]*D1 for j in N2)>=0)
    
    mb.update()

    mb.params.OutputFlag = 0
    mb.params.timelimit = 300    
    mb.optimize()
    
    zvals= mb.getAttr('x', zvars)

    return mb.objVal,zvals


def Model22(wvals,uvals,rhopara,lampara):
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
    
    zvars = mb.addVars(N,lb=0.0, ub=1.0, vtype=GRB.BINARY, name='Z')

    mb.setObjective(quicksum((1.0/float(N))*(uvals[j]-1.0)*(zvars[j]) for j in range(N))+quicksum(lampara*wvals[i]*wvals[i] for i in range(mm))
                    -quicksum(rhopara*zvars[j]/D1 for j in N1)+quicksum(rhopara*zvars[j]/D2 for j in N2), GRB.MINIMIZE)
    
    mb.update()

    mb.addConstr(quicksum(zvars[j]*D2 for j in N1)-quicksum(zvars[j]*D1 for j in N2)<=0)

    mb.update()

    mb.params.OutputFlag = 0
    mb.params.timelimit = 300    
    mb.optimize()

    zvals= mb.getAttr('x', zvars)
    
    return mb.objVal,zvals


def Model3(lampara):
    mb = Model()
    bC=1e2

    wvars = mb.addVars(mm, lb=-bC, ub=bC, vtype=GRB.CONTINUOUS, name='w')
    uvars = mb.addVars(N, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='u')
    bvar = mb.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='b')
    
    mb.setObjective(1.0/float(N)*quicksum(uvars[j] for j in range(N))+quicksum(lampara*wvars[i]*wvars[i] for i in range(mm)), GRB.MINIMIZE)
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

df = pd.DataFrame(columns=('lambda','rho','time','obj'))

title="wine_55"
book = xlrd.open_workbook(title+".xlsx")
sh= book.sheet_by_name(title)

y =[]
X =[]
i = 0
while True:
    try:
        y.append(sh.cell_value(i, 0))            
        j=0
        rows=[0]*12
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
np.random.seed(1)
iter1=0
start = time.time()
lampara=1.0
for rhopara in [0.01,0.03,0.05,0.1,0.2,0.5,0.8,1.0,2.0,3.0,5.0,10.0]:        
    maxiter =50
    uvals=[1.0]*N
    zvals=[1.0]*N
    preobj=1e25
   
    bestobj,uvals,wvals,bval=Model3(lampara)

    for j in range(N):
        if uvals[j]>=1.0:
            zvals[j]=0.0
   
    bestobj,uvals,wvals,bval=Model1(zvals,rhopara,lampara)    
   
    iterat1=0
    currentobj=1e20
    while iterat1<1:
        iteration=0
        bestobj=1e20
        while iteration<=maxiter and (math.fabs(1.0-bestobj/preobj)>0.01) and (time.time()-start<=3600):##training
            preobj=bestobj
            bestobj1,zvals1=Model21(wvals,uvals,rhopara,lampara)
            bestobj2,zvals2=Model22(wvals,uvals,rhopara,lampara)

            if bestobj1<=bestobj2:
                bestobj=bestobj1
                zvals=zvals1
            else:
                bestobj=bestobj2
                zvals=zvals2

            bestobj,uvals,wvals,bval=Model1(zvals,rhopara,lampara)
     
            iteration=iteration+1        
        iterat1=iterat1+1
        if currentobj>bestobj:
            currentobj=bestobj
            bestw=wvals
            bestb=bval
           
    totaltime=time.time()-start

    df.loc[iter1] =np.array([lampara,rhopara,totaltime,currentobj])    
    df.to_csv(title+"_IRS.csv")

    iter1=iter1+1
    

