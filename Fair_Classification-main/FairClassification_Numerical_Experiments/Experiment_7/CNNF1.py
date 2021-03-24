import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, AveragePooling2D
from keras.utils import to_categorical
from keras.optimizers import SGD
import keras.backend as K
from gurobipy import *
import math
import numpy as np
import xlrd 
import sys
import datetime
from random import sample
import scipy as sp
import pandas as pd
from sympy import *
import joblib
from sklearn import cluster
import time
import matplotlib.pyplot as plt
import warnings
from sklearn import linear_model
warnings.filterwarnings("ignore")
from numpy import *
import datetime
from read_data import imagetoarray


def Model2(predicty_train,tval,rhopara):
    mb = Model()
    bC=1e2

    S1=C5_train
    S2=C6_train
    N=len(y5)+len(y6)
    
    avars = mb.addVar( lb=0,vtype=GRB.CONTINUOUS, name='A')    
    zvars0 = mb.addVars(N,lb=0.0, ub=1.0, vtype=GRB.BINARY, name='Z0')
    zvars1 = mb.addVars(N,lb=0.0, ub=1.0, vtype=GRB.BINARY, name='Z1')
    
    mb.setObjective(quicksum(1.0/len(y5)*(tval-predicty_train[j][0])*(zvars0[j]) for j in S1)+quicksum(1.0/len(y6)*(tval-predicty_train[j][1])*(zvars1[j]) for j in S2)
                    +rhopara*(quicksum((1.0-zvars0[j]) for j in S1)+quicksum((1.0-zvars1[j]) for j in S2))*avars, GRB.MINIMIZE)
    
    mb.update()

    mb.addConstr(avars*(float(N)+quicksum(zvars0[j] for j in S1)-quicksum(zvars1[j] for j in S2))<=1.0)
    mb.addConstr(avars*(float(N)+quicksum(zvars0[j] for j in S1)-quicksum(zvars1[j] for j in S2))>=1.0)
    
    mb.update()

    mb.params.OutputFlag = 0
    mb.params.timelimit = 300
    mb.optimize()
    
    zvals0= mb.getAttr('x', zvars0)
    zvals1= mb.getAttr('x', zvars1)

    return mb.objVal,zvals0,zvals1

    
global df


# #Race 
# filename1 = '20s_Asian' 
# filename2 = '20s_White' 
# filename3 = '60s_Asian' 
# filename4 = '60s_White' 
# (x1, vax1) = imagetoarray(filename1, 31, 22) 
# y1 = np.ones(22)
# vay1 = np.ones(9)
# (x2, vax2) = imagetoarray(filename3, 27, 19) 
# y2 = np.ones(19)
# vay2 = np.ones(8)
# (x3, vax3) = imagetoarray(filename2, 110, 77) 
# y3 = np.ones(77) + 1
# vay3 = np.ones(33)+ 1
# (x4, vax4) = imagetoarray(filename4, 101, 70) 
# y4 = np.ones(70) + 1
# vay4 = np.ones(31) + 1

# #X-ray Infiltration 
# filename1 = 'Infiltration_female' 
# filename3 = 'Infiltration_male' 
# filename2 = 'NoInfiltration_female' 
# filename4 = 'NoInfiltration_male' 
# (x1, vax1) = imagetoarray(filename1, 30, 21) 
# y1 = np.ones(21)
# vay1 = np.ones(9)
# (x2, vax2) = imagetoarray(filename3, 30, 21) 
# y2 = np.ones(21) 
# vay2 = np.ones(9)
# (x3, vax3) = imagetoarray(filename2, 92, 64) 
# y3 = np.ones(64)+ 1
# vay3 = np.ones(28)+ 1
# (x4, vax4) = imagetoarray(filename4, 94, 66) 
# y4 = np.ones(66) + 1
# vay4 = np.ones(28) + 1


# #X-ray Atelec 
# filename1 = 'Atelec_female' 
# filename3 = 'Atelec_male' 
# filename2 = 'NoAtelec_female' 
# filename4 = 'NoAtelec_male'        
# (x1, vax1) = imagetoarray(filename1, 50, 35) 
# y1 = np.ones(35)
# vay1 = np.ones(15)
# (x2, vax2) = imagetoarray(filename3, 50, 35) 
# y2 = np.ones(35) 
# vay2 = np.ones(15)
# (x3, vax3) = imagetoarray(filename2, 282, 197) 
# y3 = np.ones(197)+ 1
# vay3 = np.ones(85)+ 1
# (x4, vax4) = imagetoarray(filename4, 271, 190) 
# y4 = np.ones(190) + 1
# vay4 = np.ones(81) + 1
            
#Age
filename1 = '20s_Asian' 
filename3 = '20s_White' 
filename2 = '60s_Asian' 
filename4 = '60s_White' 
(x1, vax1) = imagetoarray(filename2, 30, 21) 
y1 = np.ones(21)
vay1 = np.ones(9)
(x2, vax2) = imagetoarray(filename4, 30, 21) 
y2 = np.ones(21)
vay2 = np.ones(9) 
(x3, vax3) = imagetoarray(filename1, 104, 73) 
y3 = np.ones(73)+ 1
vay3 = np.ones(31)+ 1
(x4, vax4) = imagetoarray(filename3, 110, 77) 
y4 = np.ones(77) + 1
vay4 = np.ones(33)+ 1

# #Gender
# filename1 = 'Female_5' 
# filename3 = 'Female_80' 
# filename2 = 'Male_5' 
# filename4 = 'Male_80' 
# (x1, vax1) = imagetoarray(filename2, 40, 28) 
# y1 = np.ones(28)
# vay1 = np.ones(12)
# (x2, vax2) = imagetoarray(filename4, 40, 28) 
# y2 = np.ones(28) 
# vay2 = np.ones(12) 
# (x3, vax3) = imagetoarray(filename1, 176, 123) 
# y3 = np.ones(123)+ 1
# vay3 = np.ones(53)+ 1
# (x4, vax4) = imagetoarray(filename3, 135, 94) 
# y4 = np.ones(94) + 1
# vay4 = np.ones(41)+ 1


x5 = np.concatenate((x1, x2), axis=0)
x6 = np.concatenate((x3, x4), axis=0)
y5 = np.concatenate((y1, y2), axis=0)
y6 = np.concatenate((y3, y4), axis=0)
vax5 = np.concatenate((vax1, vax2), axis=0)
vax6 = np.concatenate((vax3, vax4), axis=0)
vay5 = np.concatenate((vay1, vay2), axis=0)
vay6 = np.concatenate((vay3, vay4), axis=0)

pixel = 50
num_classes = 2
x = np.r_[x5, x6]
vax = np.r_[vax5, vax6]
y = np.r_[y5, y6]
vay = np.r_[vay5, vay6]
y = y.astype(np.int16)
vay = vay.astype(np.int16)
x = x/255
vax = vax/255
y = y - 1
vay = vay - 1
y0 = to_categorical(y)    
vay0 = to_categorical(vay) 
x_train0 = x.reshape(-1, pixel, pixel, 1) 
x_test0 = vax.reshape(-1, pixel, pixel, 1) 

Ntrain=len(y5)+len(y6)
C5_train=[]
C6_train=[]
i=0
while i<=len(y5)-1:
    C5_train.append(i)
    i=i+1
j=len(y5)
while j<=len(y5)+len(y6)-1:
    C6_train.append(j)
    j=j+1
    
Ntest=len(vay5)+len(vay6)
C5_test=[]
C6_test=[]
i=0
while i<=len(vay5)-1:
    C5_test.append(i)
    i=i+1
j=len(vay5)
while j<=len(vay5)+len(vay6)-1:
    C6_test.append(j)
    j=j+1

C5=[]
C6=[]
k=0
while k<=len(y5)-1:
    C5.append(k)
    k=k+1
k=0
while k<=len(y6)-1:
    C6.append(k)
    k=k+1
              
x001=x1
vax001=vax1
y001=y1
vay001=vay1
x002=x2
vax002=vax2
y002=y2
vay002=vay2
x003=x3
vax003=vax3
y003=y3
vay003=vay3
x004=x4
vax004=vax4
y004=y4
vay004=vay4
x005=x5
vax005=vax5
y005=y5
vay005=vay5
x006=x6
vax006=vax6
y006=y6
vay006=vay6
y000=y0   
vay000=vay0 
x_train000 =x_train0 
x_test000 =x_test0 
Ntrain00=Ntrain
Ntest00=Ntest
C5_train00=C5_train
C6_train00=C6_train
C5_test00=C5_test
C6_test00=C6_test
C500=C5
C600=C6
                
# title="Infiltration"
# Ntrain=172 

# title="Atelec"
# Ntrain=457

# title="Race"
# Ntrain= 188

title="Age"
Ntrain= 192

# title="Gender"
# Ntrain= 273

df = pd.DataFrame(columns=('ks','h','C_count','ftime','testFairness','testAcc','trainFairness','trainAcc','rhopara','t'))
    
iter1=0
Hcount=1

for rhopara in [0, 0.1/float(Ntrain), 0.5/float(Ntrain), 1/float(Ntrain), 5/float(Ntrain), 10/float(Ntrain)]:
    for tval in [0.1, 0.3, 0.5, 0.7, 0.9, 0.98]:
        for h in range(Hcount):
            x1=x001
            vax1=vax001
            y1=y001
            vay1=vay001
            x2=x002
            vax2=vax002
            y2=y002
            vay2=vay002           
            x3=x003
            vax3=vax003
            y3=y003
            vay3=vay003           
            x4=x004
            vax4=vax004
            y4=y004
            vay4=vay004            
            x5=x005
            vax5=vax005
            y5=y005
            vay5=vay005            
            x6=x006
            vax6=vax006
            y6=y006
            vay6=vay006                      
            y0=y000   
            vay0=vay000
            x_train0 =x_train000
            x_test0 =x_test000
            Ntrain=Ntrain00
            Ntest=Ntest00
            C5_train=C5_train00
            C6_train=C6_train00
            C5_test=C5_test00
            C6_test=C6_test00
            C5=C500
            C6=C600
    
            N5=C5
            N6=C6
            x5_pre=x5
            x6_pre=x6
            C=5 
            C_count=1
            for i in range(C):
                x5_pre=x5
                x6_pre=x6
                x5_new=[]
                x6_new=[]
                y5_new=[]
                y6_new=[]
                
                for i in N5:
                    x5_new.append(x5_pre[i])          
                y5_new=np.ones(len(N5))
                for i in N6:
                    x6_new.append(x6_pre[i])        
                y6_new=np.ones(len(N6))+ 1
                
                N = len(y5_new) + len(y6_new)
                num_classes = 2
                x = np.r_[x5_new, x6_new]
                y = np.r_[y5_new, y6_new]
                y = y.astype(np.int16)
                x = x/255
                y = y - 1
                y = to_categorical(y)    
                x_train = x.reshape(-1, pixel, pixel, 1) 
                
                
                ks = 3
                epochs = 20                            
                start = datetime.datetime.now()
                model = Sequential()
                model.add(Conv2D(filters = 20, kernel_size=(int(ks),int(ks)),activation='relu',input_shape=(pixel, pixel, 1)))
                model.add(MaxPooling2D(pool_size=(2,2), strides = 2, padding = 'same'))
                model.add(Conv2D(filters = 50, kernel_size=(int(ks),int(ks)), activation='relu', padding = 'same'))
                model.add(MaxPooling2D(pool_size=(2,2), strides = 2, padding = 'same'))
                model.add(Flatten())
                model.add(Dense(500, activation='relu'))
                model.add(Dense(num_classes, activation='softmax'))
                sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov=True)
                
                model.compile(loss=keras.losses.BinaryCrossentropy(),optimizer='Adam',metrics=['accuracy'])
                hist = model.fit(np.array(np.atleast_3d(x_train)), y,
                          batch_size = 32,
                          epochs = epochs,
                          verbose = 0)
                end = datetime.datetime.now()
                ftime= (end - start).total_seconds()
              
                predicty=model.predict(x_train)
                score_train = model.evaluate(np.array(np.atleast_3d(x_train0)), y0, verbose = 0)
                score = model.evaluate(np.array(np.atleast_3d(x_test0)), vay0, verbose = 0)
                predicty_train=model.predict(x_train0)
                predicty_test=model.predict(x_test0)
                
                ztrain0=[]
                ztrain1=[]
                v00=0.0
                i=0
                while i <= len(y5)-1:
                    v0=1.0
                    ztrain1.append(v00)
                    if predicty_train[i][0]>=predicty_train[i][1]:
                        ztrain0.append(v0)
                    else:
                        ztrain0.append(v00)
                    i=i+1
                
                j=len(y5)
                while j <= len(y5)+len(y6)-1:
                    v1=1.0
                    ztrain0.append(v00)
                    if predicty_train[j][1]>=predicty_train[j][0]:
                        ztrain1.append(v1)
                    else: 
                        ztrain1.append(v00)
                    j=j+1
                    
                uexpr11=0.0
                for j in range(len(ztrain0)):
                    uexpr11=uexpr11+ztrain0[j]                    
                uexpr12=0.0
                for j in range(len(ztrain1)):
                    uexpr12=uexpr12+ztrain1[j]
                
                Ntrain=len(y0)
                fairness_train=(Ntrain-uexpr11-uexpr12)/(Ntrain+uexpr11-uexpr12)
                ztest0=[]
                ztest1=[]
                i=0
                while i <= len(vay5)-1:
                    v0=1.0
                    ztest1.append(v00)
                    if predicty_test[i][0]>=predicty_test[i][1]:
                        ztest0.append(v0)
                    else:
                        ztest0.append(v00)
                    i=i+1
                
                j=len(vay5)
                while j <= len(vay5)+len(vay6)-1:
                    v1=1.0                            
                    ztest0.append(v00)
                    if predicty_test[j][1]>=predicty_test[j][0]:
                        ztest1.append(v1)
                    else:
                        ztest1.append(v00)
                    j=j+1
                                  
                uexpr11=0.0
                for j in range(len(ztest0)):
                    uexpr11=uexpr11+ztest0[j]
                uexpr12=0.0
                for j in range(len(ztest1)):
                    uexpr12=uexpr12+ztest1[j]
                
                Ntest=len(vay0)
                fairness_test=(Ntest-uexpr11-uexpr12)/(Ntest+uexpr11-uexpr12)
                
                df.loc[iter1] =np.array([ks,h,C_count,ftime,fairness_test,score[1],fairness_train,score_train[1],rhopara,tval])
                df.to_csv(title+"_CNNF1.csv")
    
                iter1=iter1+1
                
                bestobj,zvals0,zvals1=Model2(predicty_train,tval,rhopara)
                
                N5=[]
                N6=[]            
                for i in C5_train: 
                    if zvals0[i]==1:
                        N5.append(i)   
                for i in C6_train:
                    if zvals1[i]==1:
                        N6.append(i-len(y))    
                        
                C_count=C_count+1
                    
                          
