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
import scipy as sp
import pandas as pd
import sys
import datetime
from random import sample
from sympy import *
import joblib
import time
import warnings
warnings.filterwarnings("ignore")
from numpy import *
import datetime
from read_data import imagetoarray

def Model21(predicty_train,tval,rhopara):
    mb = Model()
    bC=1e2

    zvar0 = mb.addVars(Ntrain,lb=0.0, ub=1.0, vtype=GRB.BINARY, name='Z1')
    zvar1 = mb.addVars(Ntrain,lb=0.0, ub=1.0, vtype=GRB.BINARY, name='Z2')
                
    mb.setObjective(quicksum((1.0/float(Ntrain))*(tval-predicty_train[j][0])*(zvar0[j]) for j in C5)+quicksum((1.0/float(Ntrain))*(tval-predicty_train[j][1])*(zvar1[j]) for j in C6)
                    +quicksum(rhopara*zvar0[j]/D1 for j in N11)+quicksum(rhopara*zvar1[j]/D1 for j in N12)-quicksum(rhopara*zvar0[j]/D2 for j in N21)-quicksum(rhopara*zvar1[j]/D2 for j in N22), GRB.MINIMIZE)

    mb.update()

    mb.addConstr(quicksum(zvar0[j]*D2 for j in N11)+quicksum(zvar1[j]*D2 for j in N12)-quicksum(zvar0[j]*D1 for j in N21)-quicksum(zvar1[j]*D1 for j in N22)>=0)

    mb.update()

    mb.params.OutputFlag = 0
    mb.params.timelimit = 300
    mb.optimize()
    
    zvals0= mb.getAttr('x', zvar0)
    zvals1= mb.getAttr('x', zvar1)
    
    return mb.objVal,zvals0,zvals1


def Model22(predicty_train,tval,rhopara):
    mb = Model()
    bC=1e2

    zvar0 = mb.addVars(Ntrain,lb=0.0, ub=1.0, vtype=GRB.BINARY, name='Z1')
    zvar1 = mb.addVars(Ntrain,lb=0.0, ub=1.0, vtype=GRB.BINARY, name='Z2')
                      
    mb.setObjective(quicksum((1.0/float(Ntrain))*(tval-predicty_train[j][0])*(zvar0[j]) for j in C5)+quicksum((1.0/float(Ntrain))*(tval-predicty_train[j][1])*(zvar1[j]) for j in C6)    
                    -quicksum(rhopara*zvar0[j]/D1 for j in N11)-quicksum(rhopara*zvar1[j]/D1 for j in N12)+quicksum(rhopara*zvar0[j]/D2 for j in N21)+quicksum(rhopara*zvar1[j]/D2 for j in N22), GRB.MINIMIZE)
                
    mb.update()

    mb.addConstr(quicksum(zvar0[j]*D2 for j in N11)+quicksum(zvar1[j]*D2 for j in N12)-quicksum(zvar0[j]*D1 for j in N21)-quicksum(zvar1[j]*D1 for j in N22)<=0)

    mb.update()
    
    mb.params.OutputFlag = 0
    mb.params.timelimit = 300    
    mb.optimize()
    
    zvals0= mb.getAttr('x', zvar0)
    zvals1= mb.getAttr('x', zvar1)
    
    return mb.objVal,zvals0,zvals1



# Gender
filename1 = 'Female_5' 
filename3 = 'Female_80' 
filename2 = 'Male_5' 
filename4 = 'Male_80' 
(x1, vax1) = imagetoarray(filename1, 176, 123) 
y1 = np.ones(123)
vay1 = np.ones(53)
(x2, vax2) = imagetoarray(filename3, 135, 94) 
y2 = np.ones(94) 
vay2 = np.ones(41)
(x3, vax3) = imagetoarray(filename2, 161, 112) 
y3 = np.ones(112)+ 1
vay3 = np.ones(49)+ 1
(x4, vax4) = imagetoarray(filename4, 157, 109) 
y4 = np.ones(109) + 1
vay4 = np.ones(48) + 1

# #Age
# filename1 = '20s_Asian' 
# filename3 = '20s_White' 
# filename2 = '60s_Asian' 
# filename4 = '60s_White' 
# (x1, vax1) = imagetoarray(filename1, 104, 73) 
# y1 = np.ones(73)
# vay1 = np.ones(31)
# (x2, vax2) = imagetoarray(filename3, 110, 77) 
# y2 = np.ones(77) 
# vay2 = np.ones(33)
# (x3, vax3) = imagetoarray(filename2, 90, 63) 
# y3 = np.ones(63)+ 1
# vay3 = np.ones(27)+ 1
# (x4, vax4) = imagetoarray(filename4, 101, 70) 
# y4 = np.ones(70) + 1
# vay4 = np.ones(31) + 1

# #Race
# filename1 = '20s_Asian' 
# filename2 = '20s_White' 
# filename3 = '60s_Asian' 
# filename4 = '60s_White' 
# (x1, vax1) = imagetoarray(filename1, 104, 73) 
# y1 = np.ones(73)
# vay1 = np.ones(31)
# (x2, vax2) = imagetoarray(filename3, 90, 63) 
# y2 = np.ones(63)
# vay2 = np.ones(27)
# (x3, vax3) = imagetoarray(filename2, 110, 77) 
# y3 = np.ones(77) + 1
# vay3 = np.ones(33)+ 1
# (x4, vax4) = imagetoarray(filename4, 101, 70) 
# y4 = np.ones(70) + 1
# vay4 = np.ones(31) + 1

# #X-ray Atelec
# filename1 = 'Atelec_female' 
# filename3 = 'Atelec_male' 
# filename2 = 'NoAtelec_female' 
# filename4 = 'NoAtelec_male' 
# (x1, vax1) = imagetoarray(filename1, 276, 193) 
# y1 = np.ones(193)
# vay1 = np.ones(83)
# (x2, vax2) = imagetoarray(filename3, 293, 205) 
# y2 = np.ones(205) 
# vay2 = np.ones(88)
# (x3, vax3) = imagetoarray(filename2, 282, 197) 
# y3 = np.ones(197)+ 1
# vay3 = np.ones(85)+ 1
# (x4, vax4) = imagetoarray(filename4, 271, 190) 
# y4 = np.ones(190) + 1
# vay4 = np.ones(81) + 1

# #X-ray Infiltration 
# filename1 = 'Infiltration_female' 
# filename3 = 'Infiltration_male' 
# filename2 = 'NoInfiltration_female' 
# filename4 = 'NoInfiltration_male' 
# (x1, vax1) = imagetoarray(filename1, 92, 64) 
# y1 = np.ones(64)
# vay1 = np.ones(28)
# (x2, vax2) = imagetoarray(filename3, 90, 63) 
# y2 = np.ones(63) 
# vay2 = np.ones(27)
# (x3, vax3) = imagetoarray(filename2, 92, 64) 
# y3 = np.ones(64)+ 1
# vay3 = np.ones(28)+ 1
# (x4, vax4) = imagetoarray(filename4, 94, 66) 
# y4 = np.ones(66) + 1
# vay4 = np.ones(28) + 1

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
    
G1_train=[]
G2_train=[]  
for i in range(len(y0)):
    if i <= len(y1)-1 or (i >= len(y1)+len(y2) and i <= len(y1)+len(y2)+len(y3)-1): 
        G1_train.append(i)
    else:
        G2_train.append(i)

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

G1_test=[]
G2_test=[]      
for i in range(len(vay0)):
    if i <= len(vay1)-1 or (i >= len(vay1)+len(vay2) and i <= len(vay1)+len(vay2)+len(vay3)-1): 
        G1_test.append(i)
    else:
        G2_test.append(i)

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
# Ntrain=257

# title="Atelec"
# Ntrain=785

# title="Race"
# Ntrain= 283

# title="Age"
# Ntrain= 283

title="Gender"
Ntrain= 438

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
                    
                D1_train=len(G1_train)
                D2_train=len(G2_train)
                zval1_train=sum(ztrain0[i] for i in G1_train)+sum(ztrain1[i] for i in G1_train)
                zval2_train=sum(ztrain0[i] for i in G2_train)+sum(ztrain1[i] for i in G2_train)                              
                fairness_train=math.fabs(zval1_train/D1_train-zval2_train/D2_train)
                
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
              
                D1_test=len(G1_test)
                D2_test=len(G2_test)
                zval1_test=sum(ztest0[i] for i in G1_test)+sum(ztest1[i] for i in G1_test)
                zval2_test=sum(ztest0[i] for i in G2_test)+sum(ztest1[i] for i in G2_test)
                fairness_test=math.fabs(zval1_test/D1_test-zval2_test/D2_test)
                
                df.loc[iter1] =np.array([ks,h,C_count,ftime,fairness_test,score[1],fairness_train,score_train[1],rhopara,tval])
                df.to_csv(title+"_FCNN.csv")
    
                iter1=iter1+1
                
                N1=[]
                N2=[]  
                N11=[]
                N12=[]  
                N21=[]
                N22=[]  
                for i in range(len(y)):
                    if i <= len(y1)-1 or (i >= len(y1)+len(y2) and i <= len(y1)+len(y2)+len(y3)-1): 
                        N1.append(i)
                    else:
                        N2.append(i)
                for i in range(len(y)):
                    if i <= len(y1)-1: 
                        N11.append(i)
                    if (i >= len(y1)+len(y2) and i <= len(y1)+len(y2)+len(y3)-1): 
                        N12.append(i)
                    if (i >= len(y1) and i <= len(y1)+len(y2)-1 ):
                        N21.append(i)
                    if i >= len(y1)+len(y2)+len(y3):
                        N22.append(i)
                    
                D1=len(N1)
                D2=len(N2)
                bestobj1,zvals10,zvals11=Model21(predicty_train,tval,rhopara)
                bestobj2,zvals20,zvals21=Model22(predicty_train,tval,rhopara)
            
                if bestobj1<=bestobj2:
                    bestobj=bestobj1
                    zvals0=zvals10
                    zvals1=zvals11
                else:
                    bestobj=bestobj2
                    zvals0=zvals20
                    zvals1=zvals21
                                 
                N5=[]
                N6=[]
                for i in C5_train: 
                    if zvals0[i]==1:
                        N5.append(i)   
                for i in C6_train:
                    if zvals1[i]==1:
                        N6.append(i-len(y5))
                        
                C_count=C_count+1
                    
            
          
