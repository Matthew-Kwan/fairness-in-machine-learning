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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def Model1(zvals1,zvals2,rhopara,lampara,tpara,N00):
    mb = Model()
    bC=1e2
    
    C1=[]
    C2=[]
    L1=0
    L2=0
    for j in range(N00):
        if y_train[j]==1:
            C1.append(j)
            L1=L1+1
        else:        
            C2.append(j)
            L2=L2+1
                   
    wvars = mb.addVars(mm, lb=-bC, ub=bC, vtype=GRB.CONTINUOUS, name='w')
    uvars1 = mb.addVars(N00, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='u1')
    uvars2 = mb.addVars(N00, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='u2')
    bvar = mb.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='b')
    
    mb.setObjective(1.0/float(N00)*quicksum((uvars1[j]-tpara)*zvals1[j] for j in C1)+1.0/float(N00)*quicksum((uvars2[j]-tpara)*(1-zvals2[j]) for j in C2)
                    +quicksum(lampara*wvars[i]*wvars[i] for i in range(mm)), GRB.MINIMIZE)
    mb.update()

    for j in C1:
        ind=y_train[j]
        mb.addConstr(quicksum(ind*wvars[i]*X_train[j][i] for i in range(mm))+ind*bvar>=1.0-uvars1[j], 'w b M'+str(j)+str(ind))

    for j in C2:
        ind=y_train[j]
        mb.addConstr(quicksum(ind*wvars[i]*X_train[j][i] for i in range(mm))+ind*bvar>=1.0-uvars2[j], 'w b M'+str(j)+str(ind))

    mb.update()

    mb.params.OutputFlag = 0
    mb.params.timelimit = 300 
    mb.optimize()
    
    wvals= mb.getAttr('x', wvars)
    bval=bvar.x
    uvals1= mb.getAttr('x', uvars1)
    uvals2= mb.getAttr('x', uvars2)

    D1=0.0
    D2=0.0
    expr11=0.0
    expr12=0.0
    expr21=0.0
    expr22=0.0
    for j in range(N00):
        if X_train[j][0]==1:
            D1=D1+1.0
            if y_train[j]==1:
                expr11=expr11+zvals1[j]
            else:
                expr12=expr12+zvals2[j]
        else:        
            D2=D2+1.0
            if y_train[j]==1:
                expr21=expr21+zvals1[j]
            else:
                expr22=expr22+zvals2[j]
        
    return mb.objVal+rhopara*math.fabs((expr11+expr12)/D1-(expr21+expr22)/D2),uvals1,uvals2,wvals,bval


def Model21(uvals1,uvals2,wvals,rhopara,lampara,tpara,N00):
    mb = Model()
    bC=1e2

    D1=0.0
    D2=0.0
    N11=[]
    N12=[]
    N21=[]
    N22=[]
    for j in range(N00):
        if X_train[j][0]==1:
            D1=D1+1.0
            if y_train[j]==1:
                N11.append(j)
            else:
                N12.append(j)
        else:        
            D2=D2+1.0
            if y_train[j]==1:
                N21.append(j)
            else:
                N22.append(j)
                
    C1=[]
    C2=[]
    L1=0
    L2=0
    for j in range(N00):
        if y_train[j]==1:
            C1.append(j)
            L1=L1+1
        else:        
            C2.append(j)
            L2=L2+1
                
    zvars1 = mb.addVars(N00,lb=0.0, ub=1.0, vtype=GRB.BINARY, name='Z')
    zvars2 = mb.addVars(N00,lb=0.0, ub=1.0, vtype=GRB.BINARY, name='Z')

    mb.setObjective(1.0/float(N00)*quicksum((uvals1[j]-tpara)*zvars1[j] for j in C1)+1.0/float(N00)*quicksum((uvals2[j]-tpara)*(1-zvars2[j]) for j in C2)
                    +quicksum(lampara*wvals[i]*wvals[i] for i in range(mm))
                    +(quicksum(rhopara*zvars1[j] for j in N11)+quicksum(rhopara*zvars2[j] for j in N12))/D1
                    -(quicksum(rhopara*zvars1[j] for j in N21)+quicksum(rhopara*zvars2[j] for j in N22))/D2, GRB.MINIMIZE)
                
    mb.update()

    mb.addConstr((quicksum(rhopara*zvars1[j] for j in N11)+quicksum(rhopara*zvars2[j] for j in N12))*D2
                 -(quicksum(rhopara*zvars1[j] for j in N21)+quicksum(rhopara*zvars2[j] for j in N22))*D1>=0)

    mb.update()

    mb.params.OutputFlag = 0
    mb.params.timelimit = 300
    mb.optimize()
    
    zvals1= mb.getAttr('x', zvars1)
    zvals2= mb.getAttr('x', zvars2)
    
    return mb.objVal,zvals1,zvals2


def Model22(uvals1,uvals2,wvals,rhopara,lampara,tpara,N00):
    mb = Model()
    bC=1e2

    D1=0.0
    D2=0.0
    N11=[]
    N12=[]
    N21=[]
    N22=[]
    for j in range(N00):
        if X_train[j][0]==1:
            D1=D1+1.0
            if y_train[j]==1:
                N11.append(j)
            else:
                N12.append(j)
        else:        
            D2=D2+1.0
            if y_train[j]==1:
                N21.append(j)
            else:
                N22.append(j)
    C1=[]
    C2=[]
    L1=0
    L2=0   
    for j in range(N00):
        if y_train[j]==1:
            C1.append(j)
            L1=L1+1
        else:        
            C2.append(j)
            L2=L2+1
                
    zvars1 = mb.addVars(N00,lb=0.0, ub=1.0, vtype=GRB.BINARY, name='Z')
    zvars2 = mb.addVars(N00,lb=0.0, ub=1.0, vtype=GRB.BINARY, name='Z')

    mb.setObjective(1.0/float(N00)*quicksum((uvals1[j]-tpara)*zvars1[j] for j in C1)+1.0/float(N00)*quicksum((uvals2[j]-tpara)*(1-zvars2[j]) for j in C2)
                    +quicksum(lampara*wvals[i]*wvals[i] for i in range(mm))
                    -(quicksum(rhopara*zvars1[j] for j in N11)+quicksum(rhopara*zvars2[j] for j in N12))/D1
                    +(quicksum(rhopara*zvars1[j] for j in N21)+quicksum(rhopara*zvars2[j] for j in N22))/D2, GRB.MINIMIZE)
                
    mb.update()

    mb.addConstr((quicksum(rhopara*zvars1[j] for j in N11)+quicksum(rhopara*zvars2[j] for j in N12))*D2
                 -(quicksum(rhopara*zvars1[j] for j in N21)+quicksum(rhopara*zvars2[j] for j in N22))*D1<=0)

    mb.update()

    mb.params.OutputFlag = 0
    mb.params.timelimit = 300
    mb.optimize()
    
    zvals1= mb.getAttr('x', zvars1)
    zvals2= mb.getAttr('x', zvars2)
    
    return mb.objVal,zvals1,zvals2


def Model3(lampara,N00):
    mb = Model()
    bC=1e2

    wvars = mb.addVars(mm, lb=-bC, ub=bC, vtype=GRB.CONTINUOUS, name='w')
    uvars = mb.addVars(N00, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='u')
    bvar = mb.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='b')
    
    mb.setObjective(1.0/float(N00)*quicksum(uvars[j] for j in range(N00))+quicksum(lampara*wvars[i]*wvars[i] for i in range(mm)), GRB.MINIMIZE)
    
    mb.update()

    for j in range(N00):     
        ind=y_train[j]
        mb.addConstr(quicksum(ind*wvars[i]*X_train[j][i] for i in range(mm))+ind*bvar>=1.0-uvars[j], 'w b M'+str(j)+str(ind))

    mb.update()

    mb.params.OutputFlag = 0
    mb.params.timelimit = 300
    mb.optimize()
    
    wvals= mb.getAttr('x', wvars)
    bval=bvar.x
    uvals= mb.getAttr('x', uvars)

    return mb.objVal,uvals,wvals,bval

global df


df = pd.DataFrame(columns=('b','w','time','testN','testm','testM','fairness','# of accurate prediction','accuracy',
                           'N','m','M','fairnesst','# of accurate prediction t','accuracyt','obj','rho','lambda','t'))

# title="default" #23
# title="compas" #7
# title="abalone" #8
# title="studentp" #32
title="studentm" #31

book = xlrd.open_workbook(title+".xlsx")
sh= book.sheet_by_name(title)

yAll =[]
XAll =[]
i = 0
while True:
    try:
        yAll.append(sh.cell_value(i, 0))            
        j=0
        rows=[0]*31
        while True:
            try:
                rows[int(sh.cell_value(i, 2*j+1))-1]=sh.cell_value(i, 2*j+2) 
                j=j+1
            except IndexError:
                break
            except ValueError:
                break
        XAll.append(rows)          
        i = i + 1
    except IndexError:
        break
    
X, testX, y, testy = train_test_split(XAll, yAll, test_size=0.3, random_state=42)


I=2
N=len(y)
mm=len(X[0][:])
M=mm*mm*I
testN=len(testy)
testmm=len(testX[0][:])
testM=testmm*testmm*I

np.random.seed(1)
iter1=0
start = time.time()

for rhopara in [0.0, 0.01, 0.1, 0.2, 0.5, 0.8, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]:    
    for lampara in [0, 1.0/(1000.0*float(N)), 1.0/(100.0*float(N)), 0.5/float(N), 1.0/float(N), 2.0/float(N), 10.0/float(N), 100.0/float(N), 1000.0/float(N)]:
        lampara=1.25*lampara
        for tpara in [ 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0]:

            maxiter=50
            bestrate1=0.0
            Fcv=[]
            Acv=[]
                
            kf = KFold(n_splits=5)
            for train, test in kf.split(X):                
                X_train=[]
                y_train=[]
                X_test=[]
                y_test=[]
                N001=len(train)
                N002=len(test)
               
                i=0
                num=0
                for i in range(N001):
                    num=train[i]
                    xval=X[num]
                    yval=y[num]
                    X_train.append(xval)
                    y_train.append(yval)
                    i=i+1
                    
                j=0
                num=0
                for j in range(N002):
                    num=test[j]
                    xval=X[num]
                    yval=y[num]
                    X_test.append(xval)
                    y_test.append(yval)
                    j=j+1
                
                N00=len(y_train)
                bestobj,uvals,wvals,bval=Model3(lampara,N00)
    
                C1=[]
                C2=[]
                L1=0
                L2=0
                for j in range(N00):
                    if y_train[j]==1:
                        C1.append(j)
                        L1=L1+1
                    else:        
                        C2.append(j)
                        L2=L2+1
                              
                zvals1=[1.0]*N00
                zvals2=[0.0]*N00               
                for j in C1: 
                    if uvals[j]>=1.0: 
                        zvals1[j]=0.0                
                for j in C2: 
                    if uvals[j]>=1.0:
                        zvals2[j]=1.0 
                    
                bestobj,uvals1,uvals2,wvals,bval,D1,expr1,D2,expr2=Model1(zvals1,zvals2,rhopara,lampara,tpara,N00)  
    
                iterat1=0
                currentobj=1e20
    
                while iterat1<1:
                    iteration=0
                    preobj=1e25                 
                    while iteration<=maxiter and (math.fabs(1.0-bestobj/preobj)>0.01) and (time.time()-start<=3600):
                        preobj=bestobj
                  
                        bestobj1,zvals11,zvals12=Model21(uvals1,uvals2,wvals,rhopara,lampara,tpara,N00)
                        bestobj2,zvals21,zvals22=Model22(uvals1,uvals2,wvals,rhopara,lampara,tpara,N00)
    
                        if bestobj1<=bestobj2:
                            bestobj=bestobj1
                            zvals1=zvals11
                            zvals2=zvals12
                        else:
                            bestobj=bestobj2
                            zvals1=zvals21
                            zvals2=zvals22
                        
                        bestobj,uvals1,uvals2,wvals,bval=Model1(zvals1,zvals2,rhopara,lampara,tpara,N00)
                        
                        iteration=iteration+1                   
                    iterat1=iterat1+1
                    if currentobj>bestobj:
                        currentobj=bestobj
                        bestw0=wvals
                        bestb0=bval
       
                count1=0    
                Nnew=len(y_test)
                pred1=[0]*Nnew     
                for j in range(Nnew):
                    arr1 = (sum(bestw0[i]*float(X_test[j][i]) for i in range(mm))+bestb0)
                    ind= np.sign(arr1)
                    pred1[j]=ind
                    if (y_test[j]==ind):
                        count1=count1+1
                                           
                D1=0.0
                D2=0.0
                expr11=0.0
                expr12=0.0
                expr21=0.0
                expr22=0.0           
                for j in range(Nnew):
                    if X_test[j][0]==1:
                        D1=D1+1.0
                        if y_test[j]==1:
                            expr11=expr11+zvals1[j]
                        else:
                            expr12=expr12+zvals2[j]
                    else:        
                        D2=D2+1.0
                        if y_test[j]==1:
                            expr21=expr21+zvals1[j]
                        else:
                            expr22=expr22+zvals2[j]
                                       
                Fairness1=math.fabs((expr11+expr12)/D1-(expr21+expr22)/D2)
                rate1=float(count1)*(1.0/float(Nnew)) 
              
                if bestrate1<rate1:
                    bestrate1=rate1
                    bestw=bestw0
                    bestb=bestb0
    
                Fcv.append(Fairness1)
                Acv.append(rate1)
                
            newFairness1=sum(Fcv)/len(Fcv)
            newrate1=sum(Acv)/len(Acv)
     
        
            L1=0.0
            L2=0.0
            for j in range(testN):
                if testy[j]==1:
                    L1=L1+1.0
                else:        
                    L2=L2+1.0
    
            pred1=[0]*testN
            pred2=[0]*testN
            zpred1=[0.0]*testN   
            zpred2=[1.0]*testN   
            count1=0
            count2=0
            for j in range(testN):
                if testy[j]==1:
                    arr = (sum(bestw[i]*float(testX[j][i]) for i in range(mm))+bestb)
                    ind= np.sign(arr)
                    pred1[j]=ind
                    if (testy[j]==ind):
                        count1=count1+1
                        zpred1[j]=1.0    
                else:
                    arr = (sum(bestw[i]*float(testX[j][i]) for i in range(mm))+bestb)
                    ind= np.sign(arr)
                    pred2[j]=ind
                    if (testy[j]==ind):
                        count2=count2+1
                        zpred2[j]=0.0 
                    
            D1=0.0
            D2=0.0
            expr11=0.0
            expr12=0.0
            expr21=0.0
            expr22=0.0     
            for j in range(testN):
                if testX[j][0]==1:
                    D1=D1+1.0
                    if testy[j]==1:
                        expr11=expr11+zpred1[j]
                    else:
                        expr12=expr12+zpred2[j]
                else:        
                    D2=D2+1.0
                    if testy[j]==1:
                        expr21=expr21+zpred1[j]
                    else:
                        expr22=expr22+zpred2[j]
                
            Fairness=math.fabs((expr11+expr12)/D1-(expr21+expr22)/D2)
            count=float(count1)+float(count2)
            rate=count*(1.0/float(testN)) 
           
            totaltime=time.time()-start
    
            df.loc[iter1] =np.array([str(bestb),str(bestw),totaltime,testN,testmm,testM,Fairness,count,rate,N,mm,M,newFairness1,count1,newrate1,currentobj,rhopara,lampara,tpara])         
            df.to_csv(title+"_GSVMF_DP.csv")
    
            iter1=iter1+1
    

