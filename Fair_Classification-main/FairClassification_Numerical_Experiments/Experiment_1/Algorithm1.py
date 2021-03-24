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


global df

df = pd.DataFrame(columns=('rho','time','obj'))
title="wine_5000"
book = xlrd.open_workbook(title+".xlsx")
sh= book.sheet_by_name(title)

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
np.random.seed(1)
iter1=0
start = time.time()  
for rhopara in [0.01,0.03,0.05,0.1,0.2,0.5,0.8,1.0,2.0,3.0,5.0,10.0]:
    uvals=[1.0]*N
    wvals=[1.0]*N
    r=10.0
    for j in range(N):
        uvals[j]=np.random.uniform(0,r)        
        wvals[j]=np.random.uniform(-r,r) 
    bval=100
    vk2=[]
    bestv=1e20
    D1=0.0
    D2=0.0
    N1=[]
    N2=[]
    S2=[]
    Sh1=[]
    uhat1=[]
    uhat11=[]
    v12=[]
    for j in range(N):
        if X[j][0]==1:
            D1=D1+1.0
            u=uvals[j]
            uh=(1.0/float(N))*(uvals[j]-1.0)
            Sh1.append(uh)           
            N1.append(j)
        else:        
            D2=D2+1.0
            u=uvals[j]
            S2.append(u)
            N2.append(j)
   
    Sh1.sort()    
    S2.sort()
    u1=[]
    u2=[]
    k2=0
    while k2<=D2:
        if k2>0:
            u2=S2[0:k2]
        Shat1=[]
        Shat11=[]

        k1=math.floor(k2*D1/D2)
        if k1>0:
            u1=Sh1[0:k1]
        kk=len(u1)
        tstar=0
        para=rhopara/D1
        j=0
        for j in range(kk): 
            if u1[j]<para:
                tstar=tstar+1
        k1=np.min([math.floor(k2*D1/D2),tstar])
        v1=sum(u1[j]-rhopara/D1 for j in range(k1))+sum((1.0/float(N))*(u2[j]-1.0)+rhopara/D2 for j in range(k2))
        

        k11=int(D1)  
        if k11>0:
            u11=Sh1[0:k11]
        tstar1=0
        para=-rhopara/D1
        j=0
        for j in range(k11): 
            if u11[j]<para:
                tstar1=tstar1+1
        k11=np.max([math.ceil(k2*D1/D2),tstar1])                                
        v2=sum(u11[j]+rhopara/D1 for j in range(k11))+sum((1.0/float(N))*(u2[j]-1.0)-rhopara/D2 for j in range(k2))             
                                                         
 
        vk2=min(v1,v2)
        k2=k2+1  
        if bestv>vk2:
            bestv=vk2
       
    totaltime=time.time()-start

    
    df.loc[iter1] =np.array([rhopara,totaltime,bestv])
    df.to_csv(title+"_alg1.csv")
    iter1=iter1+1
    

