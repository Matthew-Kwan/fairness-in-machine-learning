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


def Model1(zvals,rhopara,lampara,tpara):
    mb = Model()
    bC=1e2

    wvars = mb.addVars(mm, lb=-bC, ub=bC, vtype=GRB.CONTINUOUS, name='w')
    uvars = mb.addVars(N, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='u')
    bvar = mb.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='b')
    
    mb.setObjective(1.0/float(N)*quicksum((uvars[j]-tpara)*(zvals[j]) for j in range(N))+quicksum(lampara*wvars[i]*wvars[i] for i in range(mm)), GRB.MINIMIZE)
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
        if X0[j]==1:
            D1=D1+1.0
            expr1=expr1+zvals[j]
        else:        
            D2=D2+1.0
            expr2=expr2+zvals[j]
        
    return mb.objVal+rhopara*math.fabs(expr1/D1-expr2/D2),uvals,wvals,bval,D1,expr1,D2,expr2



def Model21(uvals,wvals,rhopara,lampara,tpara):
    mb = Model()
    bC=1e2

    D1=0.0
    D2=0.0
    N1=[]
    N2=[]
    for j in range(N):
        if X0[j]==1:
            D1=D1+1.0           
            N1.append(j)
        else:        
            D2=D2+1.0           
            N2.append(j)
    
    zvars = mb.addVars(N,lb=0.0, ub=1.0, vtype=GRB.BINARY, name='Z')

    mb.setObjective(quicksum((1.0/float(N))*(uvals[j]-tpara)*(zvars[j]) for j in range(N))+quicksum(lampara*wvals[i]*wvals[i] for i in range(mm))
                    +quicksum(rhopara*zvars[j]/D1 for j in N1)
                    -quicksum(rhopara*zvars[j]/D2 for j in N2), GRB.MINIMIZE)
               
    mb.update()

    mb.addConstr(quicksum(zvars[j]*D2 for j in N1)-quicksum(zvars[j]*D1 for j in N2)>=0)

    mb.update()
    mb.params.OutputFlag = 0
    mb.params.timelimit = 300    
    mb.optimize()

    zvals= mb.getAttr('x', zvars)
    
    return mb.objVal,zvals


def Model22(uvals,wvals,rhopara,lampara,tpara):
    mb = Model()    
    bC=1e2

    D1=0.0
    D2=0.0
    N1=[]
    N2=[]
    for j in range(N):
        if X0[j]==1:
            D1=D1+1.0
            N1.append(j)
        else:        
            D2=D2+1.0
            N2.append(j)
    
    zvars = mb.addVars(N,lb=0.0, ub=1.0, vtype=GRB.BINARY, name='Z')

    mb.setObjective(quicksum((1.0/float(N))*(uvals[j]-tpara)*(zvars[j]) for j in range(N))+quicksum(lampara*wvals[i]*wvals[i] for i in range(mm))
                    -quicksum(rhopara*zvars[j]/D1 for j in N1)
                    +quicksum(rhopara*zvars[j]/D2 for j in N2), GRB.MINIMIZE)

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



df = pd.DataFrame(columns=('N','m','M','b', 'w','fairness','# of accurate prediction','accuracy','obj','t','lambda','rho'))
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
M=mm*mm*I
np.random.seed(1)
iter1=0

start = time.time()

for rhopara in [0.2]:    
    for lampara in [0]:
        for tpara in [0.3]:
            
            lamsvm=0.5
            bestobj,uvals,wvals,bval=Model3(lamsvm)
    
            maxiter=50
            iteration=0
            iterat1=0
            currentobj=1e20
            while iterat1<1:
 
                iteration=0  
                preobj=1e25
               
                while iteration<=maxiter and (math.fabs(1.0-bestobj/preobj)>0.01) and (time.time()-start<=3600):##training
                    preobj=bestobj
                     
                    bestobj1,zvals1=Model21(uvals,wvals,rhopara,lampara,tpara)
                    bestobj2,zvals2=Model22(uvals,wvals,rhopara,lampara,tpara)
    
                    if bestobj1<=bestobj2:
                        bestobj=bestobj1
                        zvals=zvals1
                    else:
                        bestobj=bestobj2
                        zvals=zvals2

                    bestobj,uvals,wvals,bval,D1,expr1,D2,expr2=Model1(zvals,rhopara,lampara,tpara)
                    
                    iteration=iteration+1
                iterat1=iterat1+1
                if currentobj>bestobj:
                    currentobj=bestobj
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
                     
            expr11=0.0
            expr12=0.0
            D11=0.0
            D12=0.0
            for j in range(N):
                if X0[j]==1:
                    D11=D11+1.0
                    expr11=expr11+zpred[j]
                else:
                    D12=D12+1.0
                    expr12=expr12+zpred[j]
                     
            Fairness=math.fabs(expr11/D11-expr12/D12)
            rate=float(count)*(1.0/float(N)) 
            
            df.loc[iter1] =np.array([N,mm,M,str(bestb),str(bestw),Fairness,count,rate,currentobj,tpara,lampara,rhopara])
            df.to_csv(title+"_GSVMF.csv")
            iter1=iter1+1
        