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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def Model21(wvals,bval,rhopara,lampara,tpara):
    mb=Model()
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
                
    mb.setObjective(1.0/float(N)*quicksum(y[j]*(math.log(tpara)-math.log(1.0/(1.0+max(1e-15,math.exp(-bval-sum(wvals[i]*X[j][i] for i in range(mm)))))))*zvars[j]
                                          +(1.0-y[j])*(math.log(tpara)-math.log(1.0-1.0/(1.0+max(1e-15,math.exp(-bval-sum(wvals[i]*X[j][i] for i in range(mm)))))))*zvars[j] for j in range(N))
                    +quicksum(lampara*wvals[i]*wvals[i] for i in range(mm))+quicksum(rhopara*zvars[j]/D1 for j in N1)-quicksum(rhopara*zvars[j]/D2 for j in N2), GRB.MINIMIZE)
                         
    mb.update()

    mb.addConstr(quicksum(zvars[j]*D2 for j in N1)-quicksum(zvars[j]*D1 for j in N2)>=0)

    mb.update()

    mb.params.OutputFlag = 0
    mb.params.timelimit = 300
    mb.optimize()
    
    zvals= mb.getAttr('x', zvars)

    return mb.objVal,zvals


def Model22(wvals,bval,rhopara,lampara,tpara):
    mb=Model()
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
         
    mb.setObjective(1.0/float(N)*quicksum(y[j]*(math.log(tpara)-math.log(1.0/(1.0+max(1e-15,math.exp(-bval-sum(wvals[i]*X[j][i] for i in range(mm)))))))*zvars[j]
                                          +(1.0-y[j])*(math.log(tpara)-math.log(1.0-1.0/(1.0+max(1e-15,math.exp(-bval-sum(wvals[i]*X[j][i] for i in range(mm)))))))*zvars[j] for j in range(N))
                    +quicksum(lampara*wvals[i]*wvals[i] for i in range(mm))-quicksum(rhopara*zvars[j]/D1 for j in N1)+quicksum(rhopara*zvars[j]/D2 for j in N2), GRB.MINIMIZE)
             
    mb.update()

    mb.addConstr(quicksum(zvars[j]*D2 for j in N1)-quicksum(zvars[j]*D1 for j in N2)<=0)

    mb.update()

    mb.params.OutputFlag = 0
    mb.params.timelimit = 300
    mb.optimize()
    
    zvals= mb.getAttr('x', zvars)

    return mb.objVal,zvals


global df


df = pd.DataFrame(columns=('w','b','time','testN','testm','testM','testFairness','# of accurate prediction','testAccuracy',
                           'N','m','M','trainFairness','# of accurate prediction','trainAccuracy','obj','rho','lambda','t'))

#title="default_LR" #23
#title="compas_LR" #7
#title="abalone_LR" #8 
#title="studentp_LR" #32
title="studentm_LR" #31

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
        for tpara in [ 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0]:
            maxiter =50
            max_iter = 50
            
            clf = LogisticRegression(random_state=0).fit(X, y)
            wvals0=clf.coef_
            wvals1 = wvals0.tolist()
            wvals = sum(wvals1, [])            
            bval0=clf.intercept_
            bval=bval0
    
            iterat1=0
            currentobj=1e20
            while iterat1<1:
                iteration=0
                bestobj=1e20
                preobj=1e25
                ratio_obj=0.0
                while iteration<=maxiter and (math.fabs(1.0-ratio_obj)>0.01) and (time.time()-start<=3600):##training
                    preobj=bestobj
        
                    bestobj1,zvals1=Model21(wvals,bval,rhopara,lampara,tpara)            
                    bestobj2,zvals2=Model22(wvals,bval,rhopara,lampara,tpara)
     
                    if bestobj1<=bestobj2:
                        bestobj=bestobj1
                        zvals=zvals1
                    else:
                        bestobj=bestobj2
                        zvals=zvals2
                    
                    wvars =wvals
                    bvar=bval                    
                    L=1000.0
                    for t in range(0, max_iter):               
                        for i in range(mm):
                            gradw=1.0/float(N)*sum(-zvals[j]*X[j][i]*math.exp(-bvar-sum(wvars[k]*X[j][k] for k in range(mm)))/(1.0+math.exp(-bvar-sum(wvars[k]*X[j][k] for k in range(mm))))+(1.0-y[j])*zvals[j]*X[j][i] for j in range(N))+2.0*lampara*wvars[i]  
                            wvars[i] = wvars[i] - 1.0/L*gradw
                        
                        gradb=1.0/float(N)*sum(y[j]*zvals[j]*(math.exp(-bvar-sum(wvars[k]*X[j][k] for k in range(mm)))/(1.0+math.exp(-bvar-sum(wvars[k]*X[j][k] for k in range(mm)))))+(1.0-y[j])*zvals[j]*(math.exp(-bvar-sum(wvars[k]*X[j][k] for k in range(mm)))/(1.0+math.exp(-bvar-sum(wvars[k]*X[j][k] for k in range(mm)))))-(1.0-y[j])*zvals[j] for j in range(N))                           
                        bvar = bvar - 1.0/L*gradb
                               
                        obj_val=1.0/float(N)*sum(y[j]*(math.log(tpara)-math.log(1.0/(1.0+max(1e-15,math.exp(-bvar-sum(wvars[i]*X[j][i] for i in range(mm)))))))*zvals[j]+(1.0-y[j])*(math.log(tpara)-math.log(1.0-1.0/(1.0+max(1e-15,math.exp(-bvar-sum(wvars[i]*X[j][i] for i in range(mm)))))))*zvals[j] for j in range(N))+sum(lampara*wvars[i]*wvars[i] for i in range(mm))
              
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

                    bestobj0=obj_val+rhopara*math.fabs(expr1/D1-expr2/D2)
                                            
                    bestobj=bestobj0
                    wvals=wvars
                    bval=bvar
                    if preobj==0:
                        ratio_obj=1
                    else:
                        ratio_obj=bestobj/preobj
       
                    iteration=iteration+1         
                iterat1=iterat1+1
                if currentobj>bestobj0:
                    currentobj=bestobj0
                    bestw=wvals
                    bestb=bval
                   
    
            count1=0    
            pred1=[0]*N    
            for j in range(N):
                arr1 = (sum(bestw[i]*float(X[j][i]) for i in range(mm))+bestb)
                if arr1>=0:
                    pred1[j]=1.0
                else:
                    pred1[j]=0.0
                if (y[j]==pred1[j]):
                    count1=count1+1
                    
            expr11=0.0
            expr12=0.0
            D11=0.0
            D12=0.0
            for j in range(N):
                if X[j][0]==1:
                    D11=D11+1.0
                    expr11=expr11+zvals[j]
                else:
                    D12=D12+1.0
                    expr12=expr12+zvals[j]
                         
            Fairness1=math.fabs(expr11/D11-expr12/D12)
            rate1=float(count1)*(1.0/float(N)) 
    
            count=0   
            pred=[0]*testN
            zpred=[0.0]*testN      
            for j in range(testN):
                arr = (sum(bestw[i]*float(testX[j][i]) for i in range(mm))+bestb)
                if arr>=0:
                    pred[j]=1.0
                else:
                    pred[j]=0.0
                if (testy[j]==pred[j]):
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
    
            df.loc[iter1] =np.array([str(bestw),str(bestb),totaltime,testN,testmm,testM,Fairness,count,rate,N,mm,M,Fairness1,count1,rate1,currentobj,rhopara,lampara,tpara])
            df.to_csv(title+"_GLRF_OMR.csv")
    
            iter1=iter1+1
        
