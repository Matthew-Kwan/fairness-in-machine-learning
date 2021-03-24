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
from sklearn.svm import SVC


def Model21(kval,rhopara,tpara):
    mb = Model()
    bC=1e2

    D1=0.0
    D2=0.0
    N1=[]
    N2=[]
    C1=[]
    C2=[]
    N11=[]
    N12=[]
    N21=[]
    N22=[]
    
    N=len(y)
    for j in range(N):
        if X[j][0]==1:
            D1=D1+1.0
            N1.append(j)
            if y[j]==1:  
                N11.append(j)
            else:
                N12.append(j)                
        else:        
            D2=D2+1.0
            N2.append(j)
            if y[j]==1:   
                N21.append(j)
            else:
                N22.append(j)
            
    for j in range(N):
        if y[j]==1: 
            C1.append(j)
        else:         
            C2.append(j)
             
    zvars1 = mb.addVars(N,lb=0.0, ub=1.0, vtype=GRB.BINARY, name='Z')
    zvars2 = mb.addVars(N,lb=0.0, ub=1.0, vtype=GRB.BINARY, name='Z')
    
    mb.setObjective(quicksum((1.0/float(N))*(tpara-kval[j])*(zvars1[j]) for j in C1)
                    +quicksum((1.0/float(N))*(tpara+kval[j])*(zvars2[j]) for j in C2)
                    +quicksum(rhopara*zvars1[j]/D1 for j in N11)
                    +quicksum(rhopara*zvars2[j]/D1 for j in N12)
                    -quicksum(rhopara*zvars1[j]/D2 for j in N21)
                    -quicksum(rhopara*zvars2[j]/D2 for j in N22), GRB.MINIMIZE)
                
    mb.update()

    mb.addConstr(quicksum(rhopara*zvars1[j]*D2 for j in N11)+quicksum(rhopara*zvars2[j]*D2 for j in N12)     
                 -quicksum(rhopara*zvars1[j]*D1 for j in N21)-quicksum(rhopara*zvars2[j]*D1 for j in N22)>=0)

    mb.update()

    mb.params.OutputFlag = 0
    mb.params.timelimit = 300    
    mb.optimize()

    zvals1= mb.getAttr('x', zvars1)
    zvals2= mb.getAttr('x', zvars2)
    
    return mb.objVal,zvals1,zvals2

def Model22(kval,rhopara,tpara):
    mb = Model()
    bC=1e2

    D1=0.0
    D2=0.0
    N1=[]
    N2=[]
    C1=[]
    C2=[]
    N11=[]
    N12=[]
    N21=[]
    N22=[]

    N=len(y) 
    for j in range(N):
        if X[j][0]==1:
            D1=D1+1.0
            N1.append(j)
            if y[j]==1: 
                N11.append(j)
            else:
                N12.append(j)               
        else:        
            D2=D2+1.0
            N2.append(j)
            if y[j]==1:     
                N21.append(j)
            else:
                N22.append(j)
            
    for j in range(N):
        if y[j]==1: 
            C1.append(j)
        else:         
            C2.append(j)
       
 
    zvars1 = mb.addVars(N,lb=0.0, ub=1.0, vtype=GRB.BINARY, name='Z')
    zvars2 = mb.addVars(N,lb=0.0, ub=1.0, vtype=GRB.BINARY, name='Z')
    
    mb.setObjective(quicksum((1.0/float(N))*(tpara-kval[j])*(zvars1[j]) for j in C1)
                    +quicksum((1.0/float(N))*(tpara+kval[j])*(zvars2[j]) for j in C2)
                    -quicksum(rhopara*zvars1[j]/D1 for j in N11)
                    -quicksum(rhopara*zvars2[j]/D1 for j in N12)
                    +quicksum(rhopara*zvars1[j]/D2 for j in N21)
                    +quicksum(rhopara*zvars2[j]/D2 for j in N22), GRB.MINIMIZE)
              
    mb.update()

    mb.addConstr(quicksum(rhopara*zvars1[j]*D2 for j in N11)+quicksum(rhopara*zvars2[j]*D2 for j in N12)     
                 -quicksum(rhopara*zvars1[j]*D1 for j in N21)-quicksum(rhopara*zvars2[j]*D1 for j in N22)<=0)

    mb.update()

    mb.params.OutputFlag = 0
    mb.params.timelimit = 300    
    mb.optimize()

    zvals1= mb.getAttr('x', zvars1)
    zvals2= mb.getAttr('x', zvars2)
    
    return mb.objVal,zvals1,zvals2



global df

df = pd.DataFrame(columns=('time','testN','testm','testM','testFairness','testAccuracy','N','m','M','trainFairness','trainAccuracy','obj','rho','tpara'))
                           

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
    for tpara in [ 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0]:
   
        maxiter=50
        iteration=0
        bestobj=1e20

        D1=0.0
        D2=0.0
        for j in range(len(y)):
            if X[j][0]==1:
                D1=D1+1.0                
            else:
                D2=D2+1.0
                        
        iterat1=0
        currentobj=1e20
        while iterat1<1:
            iteration=0
            preobj=1e25            
            X_new=X
            y_new=y
            zvals1=[1.0]*len(X)
            zvals2=[1.0]*len(X)
            ratio_obj=0.0
            
            while iteration<=maxiter and (math.fabs(1.0-ratio_obj)>0.01) and (time.time()-start<=3600):
                preobj=bestobj              
                X_ori=X
                y_ori=y
                counta=0
                countb=0
                Xlist=[]
                for i in range(len(y)):
                    if y[i]==1:
                        if zvals1[i]==1:
                            Xlist.append(i)
                            counta=counta+1
                    else:
                        if zvals2[i]==1:
                            Xlist.append(i)
                            countb=countb+1
                                                      
                if (counta>0 and countb>0):                                   
                    X_new=[]
                    y_new=[]
                    for i in Xlist:
                        xval=X_ori[i]
                        yval=y_ori[i]
                        X_new.append(xval)
                        y_new.append(yval)                     
                   
                    clf_new = SVC(random_state=0).fit(X_new, y_new)
                    func_all=clf_new.decision_function(X)
                    kval_new=func_all
                                  
                    bestobj1,zvals11,zvals12=Model21(kval_new,rhopara,tpara)                    
                    bestobj2,zvals21,zvals22=Model22(kval_new,rhopara,tpara)
                                            
                    if bestobj1<=bestobj2:
                        bestobj=bestobj1
                        zvals1=zvals11
                        zvals2=zvals12                        
                    else:
                        bestobj=bestobj2
                        zvals1=zvals21
                        zvals2=zvals22
                                                                       
                    ratio_obj=bestobj/preobj
                else:
                    ratio_obj=1.0
                              
                iteration=iteration+1          
            iterat1=iterat1+1
            if currentobj>bestobj:
                currentobj=bestobj            
                X_best=X_new
                y_best=y_new
                            
        clf_new = SVC(random_state=0).fit(X_best, y_best)
        
        acck_train_all=clf_new.score(X, y) 
        func_all=clf_new.decision_function(X)
        
        countk=0
        zpred=[0.0]*len(y)
        for i in range(len(y)):
            prek=np.sign(func_all[i])
            if y[i]==prek:
                zpred[i]=1.0
                countk=countk+1
                
        expr1=0.0
        expr2=0.0
        D1=0.0
        D2=0.0
        for j in range(len(y)):
            if X[j][0]==1:
                D1=D1+1.0
                expr1=expr1+zpred[j]
            else:
                D2=D2+1.0
                expr2=expr2+zpred[j]
        Fairness_train=math.fabs(expr1/D1-expr2/D2)

        acck_test=clf_new.score(testX, testy) 
        func_test=clf_new.decision_function(testX)
        
        countk=0
        zpred_test=[0.0]*len(testy)
        for i in range(len(testy)):
            prek=np.sign(func_test[i])
            if testy[i]==prek:
                zpred_test[i]=1.0
                countk=countk+1
                
        expr1=0.0
        expr2=0.0
        D1=0.0
        D2=0.0
        for j in range(len(testy)):
            if testX[j][0]==1:
                D1=D1+1.0
                expr1=expr1+zpred_test[j]
            else:
                D2=D2+1.0
                expr2=expr2+zpred_test[j]
                
        Fairness_test=math.fabs(expr1/D1-expr2/D2)
          
        totaltime=time.time()-start

        df.loc[iter1] =np.array([totaltime,testN,testmm,testM,Fairness_test,acck_test,N,mm,M,Fairness_train, acck_train_all,currentobj,rhopara,tpara])        
        df.to_csv(title+"_GKSVMF_OMR.csv")
       
        iter1=iter1+1
    


