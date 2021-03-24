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

def Model1(zvals,rhopara,lampara,tpara,N00):
    mb = Model()
    bC=1e2

    wvars = mb.addVars(mm, lb=-bC, ub=bC, vtype=GRB.CONTINUOUS, name='w')
    uvars = mb.addVars(N00, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='u')
    bvar = mb.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='b')
    
    mb.setObjective(1.0/float(N00)*quicksum((uvars[j]-tpara)*(zvals[j]) for j in range(N00))+quicksum(lampara*wvars[i]*wvars[i] for i in range(mm)), GRB.MINIMIZE)
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

    D1=0.0
    D2=0.0
    expr1=0.0
    expr2=0.0
    for j in range(N00):
        if X_train[j][0]==1:
            D1=D1+1.0
            expr1=expr1+zvals[j]
        else:        
            D2=D2+1.0
            expr2=expr2+zvals[j]
        
    return mb.objVal+rhopara*math.fabs(expr1/D1-expr2/D2),uvals,wvals,bval


def Model21(uvals,wvals,rhopara,lampara,tpara,N00):
    mb = Model()
    bC=1e2

    D1=0.0
    D2=0.0
    N1=[]
    N2=[]
    for j in range(N00):
        if X_train[j][0]==1:
            D1=D1+1.0
            N1.append(j)
        else:        
            D2=D2+1.0
            N2.append(j)
       
    zvars = mb.addVars(N00,lb=0.0, ub=1.0, vtype=GRB.BINARY, name='Z')

    mb.setObjective(quicksum((1.0/float(N00))*(uvals[j]-tpara)*(zvars[j]) for j in range(N00))+quicksum(lampara*wvals[i]*wvals[i] for i in range(mm))
                    +quicksum(rhopara*zvars[j]/D1 for j in N1)-quicksum(rhopara*zvars[j]/D2 for j in N2), GRB.MINIMIZE)

    mb.update()

    mb.addConstr(quicksum(zvars[j]*D2 for j in N1)-quicksum(zvars[j]*D1 for j in N2)>=0)

    mb.update()
    mb.params.OutputFlag = 0
    mb.params.timelimit = 300
    mb.optimize()

    zvals= mb.getAttr('x', zvars)
    
    return mb.objVal,zvals


def Model22(uvals,wvals,rhopara,lampara,tpara,N00):
    mb = Model()
    bC=1e2

    D1=0.0
    D2=0.0
    N1=[]
    N2=[]
    for j in range(N00):
        if X_train[j][0]==1:
            D1=D1+1.0
            N1.append(j)
        else:        
            D2=D2+1.0
            N2.append(j)
    
    zvars = mb.addVars(N00,lb=0.0, ub=1.0, vtype=GRB.BINARY, name='Z')

    mb.setObjective(quicksum((1.0/float(N00))*(uvals[j]-tpara)*(zvars[j]) for j in range(N00))+quicksum(lampara*wvals[i]*wvals[i] for i in range(mm))
                    -quicksum(rhopara*zvars[j]/D1 for j in N1)+quicksum(rhopara*zvars[j]/D2 for j in N2), GRB.MINIMIZE)

    mb.update()

    mb.addConstr(quicksum(zvars[j]*D2 for j in N1)-quicksum(zvars[j]*D1 for j in N2)<=0)

    mb.update()
    mb.params.OutputFlag = 0
    mb.params.timelimit = 300
    mb.optimize()

    zvals= mb.getAttr('x', zvars)
    
    return mb.objVal,zvals


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



df = pd.DataFrame(columns=('b', 'w','time','testN','testm','testM','testFairness','# of accurate prediction','testAccuracy',
                           'N','m','M','trainFairness','# of accurate prediction','trainAccuracy','obj','rho','lambda','t'))

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
    
                zvals=[1.0]*N00                
                for j in range(N00): 
                    if uvals[j]>=1.0:
                        zvals[j]=0.0
                 
                bestobj,uvals,wvals,bval,D1,expr1,D2,expr2=Model1(zvals,rhopara,lampara,tpara,N00)  
      
                iterat1=0
                currentobj=1e20
    
                while iterat1<1:
                    iteration=0          
                    preobj=1e25          
                    ratio_obj=0.0
                    while iteration<=maxiter and (math.fabs(1.0-bestobj/preobj)>0.01) and (time.time()-start<=3600):
                        preobj=bestobj
                                              
                        bestobj1,zvals1=Model21(uvals,wvals,rhopara,lampara,tpara,N00)
                        bestobj2,zvals2=Model22(uvals,wvals,rhopara,lampara,tpara,N00)
                             
                        if bestobj1<=bestobj2:
                            bestobj=bestobj1
                            zvals=zvals1
                        else:
                            bestobj=bestobj2
                            zvals=zvals2
                                  
                        bestobj,uvals,wvals,bval=Model1(zvals,rhopara,lampara,tpara,N00)
                        
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
                        
                expr11=0.0
                expr12=0.0
                D11=0.0
                D12=0.0
                for j in range(Nnew):
                    if X_test[j][0]==1:
                        D11=D11+1.0
                        expr11=expr11+zvals[j]  
                    else:
                        D12=D12+1.0
                        expr12=expr12+zvals[j]
                      
                Fairness1=math.fabs(expr11/D11-expr12/D12)
                rate1=float(count1)*(1.0/float(Nnew)) 
                
                if bestrate1<rate1:
                    bestrate1=rate1
                    bestw=bestw0
                    bestb=bestb0
                 
                Fcv.append(Fairness1)
                Acv.append(rate1)
                
            newFairness1=sum(Fcv)/len(Fcv)
            newrate1=sum(Acv)/len(Acv)

    
            count=0
            pred=[0]*testN
            zpred=[0.0]*testN      
            for j in range(testN):
                arr = (sum(bestw[i]*float(testX[j][i]) for i in range(mm))+bestb)
                ind= np.sign(arr)
                pred[j]=ind
                if (testy[j]==ind):
                    count=count+1
                    zpred[j]=1.0
                    
            expr1=0.0
            expr2=0.0
            D1=0.0
            D2=0.0
            for j in range(testN):
                if testX[j][0]==1:
                    D1=D1+1.0
                    expr1=expr1+zpred[j]
                else:
                    D2=D2+1.0
                    expr2=expr2+zpred[j]
                    
            Fairness=math.fabs(expr1/D1-expr2/D2)
            rate=float(count)*(1.0/float(testN)) 
  
            totaltime=time.time()-start
           
            df.loc[iter1] =np.array([str(bestb),str(bestw),totaltime,testN,testmm,testM,Fairness,count,rate,N,mm,M,newFairness1,count1,newrate1,currentobj,rhopara,lampara,tpara])
            df.to_csv(title+"_GSVMF_OMR.csv")
 
            iter1=iter1+1
        
