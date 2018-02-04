
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 16:44:19 2016

@author: ANMISHA 
"""
import sys
import numpy as np
import re
import random
print("ubit name: anmishar")
print("person number: 50208673")
TrainN=55601
ValidN=62612
valid = ValidN - TrainN
TestN=69623
Test = TestN - ValidN
data = np.genfromtxt("Querylevelnorm.txt",dtype=None,delimiter=None)
Y=np.zeros((TrainN,1),dtype='f8')
X=np.zeros((TrainN,46),dtype='float')
Yvalid=np.zeros((valid,1),dtype='f8')
Xvalid=np.zeros((valid,46),dtype='float')
#print("XValid size: ",Xvalid.shape)
#print("YValid size: ",Yvalid.shape)

Ytest=np.zeros((Test,1),dtype='f8')
Xtest=np.zeros((Test,46),dtype='float')

X2 = np.zeros((TrainN,46),dtype='float')
Y2 = np.zeros((TrainN,1),dtype='f8')
X2Valid = np.zeros((valid,46),dtype='float')
Y2Valid = np.zeros((valid,1),dtype='f8')
X2Test = np.zeros((Test,46),dtype='float')
Y2Test = np.zeros((Test,1),dtype='f8')

#Weight = np.zeros((1,4),dtype='float')

lamda=random.uniform(0,1)
#print("lamda ",lamda)
#------------------Obtaining X and Y matrix---------------------
for row in range(0,TrainN):
    j=0
    for prepro in range(0,48):
        if prepro==0:
            Y[row]=data[row][prepro]
        else:
            if prepro!=1:
                t=data[row][prepro]
                t=re.split('[:]',t)
                X[row][j]=t[1]
                j=j+1 

y=0
for row in range(TrainN,ValidN):
    j=0
    x=0
    
    for prepro in range(0,48):
        
        if prepro==0:
            Yvalid[x]=data[row][prepro]
            x = x+1
        else:
            if prepro!=1:
                t=data[row][prepro]
                t=re.split('[:]',t)
                Xvalid[y][j]=t[1]
                
                j=j+1 
    y = y+1
          
y1=0
for row in range(ValidN,TestN):
    j=0
    x1=0
    for prepro in range(0,48):
        if prepro==0:
            Ytest[x1]=data[row][prepro]
            x1=x1+1
        else:
            if prepro!=1:
                t=data[row][prepro]
                t=re.split('[:]',t)
                Xtest[y1][j]=t[1]
                j=j+1
    y1=y1+1
    

#LetoR(X,Y)
#s=np.var(X[:,45])
#print s
"""       
print Y

print X
"""

#M=raw_input("Enter value of M: ")
#print('M is',M)
#print len(X[1,:])
#LetoR(X,Y)

#-----------------CAlculating variance matrix-----------------

def LetoR(X, Y, Xvalid, Yvalid, Xtest, Ytest):
    
    sigma=np.zeros((46,46),dtype='float')
    for i in range(0,46):
        sigma[i][i] = np.var(X[:,i])
        #print(sigma)
        #
        #M=int(M)
    #print("Sigma ",sigma)   
    M=4
    
    mu=np.zeros((M,46))
    #mu=1/46
    """
    for k in range(0,TrainN):
        for n in range(0,4):
            Z[k,n]=X[k,n]
            """
    #print("means")
    #----------------------Generating Means----------------------
    for k in range(0,M):
        n=random.randint(0,TrainN)
        mu[k]=X[n]
        #print(mu[k])
    
    #print("mu",len(mu[:,1]),len(mu[1,:]))
    #-----------------Calculating phi matrix----------------------
    phi=np.zeros((TrainN,M),dtype='float')
    for i in range(0,TrainN):
        for j in range(0,M):
            if(j==0):
                phi[i,j]=1
            else:
                t=(X[i]-mu[j])
                #print("t",t)
                t2=np.transpose(t)
                #print("t2",t2)
                t3=sigma.dot(t)
                #print("t3",t3)
                t4=t2.dot(t3)
                #print("t4",t4)
                #t4=t*t3
                t5=float(t4/2)
                phi[i,j]=(np.exp(-t5))
    """
    print("t",t)     
    print("t2",t2)       
    print("t3",t3)
    print("t4",t4)    print("t5",t5)
    """
    #print('Choose the number of Basis Functions:', M)
    #print("phi",phi)
    #print("phi",len(phi[1,:]),len(phi[:,1]))
    #------------------Calucalting Weights--------------
    W=np.zeros((M,1),dtype='float')
    p1=np.transpose(phi)
    p2=p1.dot(phi)
    p3=np.linalg.inv(p2)
    p4=p3.dot(p1)
    W=p4.dot(Y)
    print("Weights calculated from Closed Form Solution")
    print(W)
 #   Weight[1]=W
    #--------------------Calculating Regularized weights-------
    L=np.zeros((M,M),dtype="int")
    for i in range(0,M):
        L[i,i]=1
    L=lamda*L
    p3=L+p2
    p3=np.linalg.inv(p3)
    p4=p3.dot(p1)
    WR=p4.dot(Y)
    print("Regularized Weights calculated from Closed Form Solution")
    print(WR)
  #  Weight[2]=WR
    #--------------------Calculating EW--------------------
    w1=np.transpose(W)
    w2=w1.dot(W)
    EW=float(w2/2)
    #print("EW",EW)
    #--------------------Calculating ED--------------------
    h2=0
    #print("w1,phi[1]",w1,phi[1])
    for i in range(0,TrainN):
        h=w1.dot(phi[i])
        h2+=(Y[i]-h)*(Y[i]-h)
        #print("h2",h2)
    EDW=float(h2/2)
    #print("EDW",EDW)
    #--------------------Calculating ERMS-------------------
    ER1=(2*EDW)/TrainN
    ERMS=np.sqrt(ER1)
    print("RMS Error Calculated from Closed Form Solution ")
    print(ERMS)

    #-------CAlculating EW Reg-------
    wr1=np.transpose(WR)
    wr2=wr1.dot(WR)
    EWR=float(wr2/2)
    #print("EWR",EWR)

    #----------------Calculating ED Reg----------
    h2=0
    for i in range(0,TrainN):
        h=wr1.dot(phi[i])
        h2+=(Y[i]-h)*(Y[i]-h)
        #print("h2",h2)
    EDWR=float(h2/2)
    #print("EDW",EDWR)
    
    #----------------ERMS Regularized----------------
    #Ereg=EDWR+lamda*EWR
    Ereg=EDWR
    Etemp=(2*Ereg)/TrainN
    ERMSreg=np.sqrt(Etemp)
    print("Regularized RMS Error Calculated from Closed Form Solution")
    print(ERMSreg)
    
    #------------------------Stochastic-------------
    #--------Initial weights-------------
    WS=np.zeros((M,1),dtype='float')
    WS = [[0],[0],[1],[0]]#,[0],[0],[0]]#,[0],[0]]
    w1s=np.transpose(WS)
    ph1=np.zeros((M,1),dtype='float')
    #print(w1s)
    #print(phi[0])
    #---------Initial Error-------------
    h2=0
    #print("w1s,phip",w1s,phi[0])
    for p in range(0,TrainN):
        h=w1s.dot(phi[p])
        h2+=(Y[p]-h)*(Y[p]-h)
    #print("h2n",h2)
    EDs=float((h2/2))
    #print("EDs",EDs)

    ER1=(2*EDs)/TrainN
    ERMS2i=np.sqrt(ER1)
    #print("ERMS2",ERMS2i)
    #-----------Initializing all values--------
    #dE1=ERMS2i
    WSN=WS
    #Ed1=EDs
    eta=0.5
    #eta=1
    #---------Calculating W for all iterations-----
    """t
    W(t+1)=W(t)+deltaW(t)
    deltaEd=-((Y-W(t)Trans*phi))*phi
    deltaEw=W(t)
    deltaE=deltaEd+lamda*deltaEw
    deltaW=-eta*deltaE
    
    """
    """
    for i in range(0,1000):
        x1=phi.dot(WSN)
        x2=Y-x1
        x3=p1.dot(x2)
        de=-x3/TrainN #-------deltaEd------
        #print("de",de)
    
        dw=eta*de  #-----deltaW-------
        #print("dw",dw)
        WSN=WS-dw  #-----W(t+1)-------
        h2=0
        w2s=np.transpose(WSN)
        #print("i,WSN",i,WSN)
        for j in range(0,TrainN):
            h=w2s.dot(phi[j])
            h2+=(Y[j]-h)*(Y[j]-h)
            EDNs=float(h2/2)
        #print("EDNs,EDs",EDNs,EDs)
        ER1=(2*EDNs)/TrainN
        ERMS2n=np.sqrt(ER1)#------Erms-----
        #print("ERMS2n,ERMS2i",ERMS2n,ERMS2i)
        temp=ERMS2i
        ERMS2i=ERMS2n
        WS=WSN
        d=ERMS2n-temp
        d=np.abs(d)
        if(d<=0.001):
            break
    print("Wi,Wn",WS,WSN)
    w2s=np.transpose(WSN)
    print("ERMS2n,ERMS2i",ERMS2n,temp)    
    """
    #print("i:",i)
   # Weight[3]=WSN
    for x in range(0,100):
        ph=phi[x]
        for i in range(0,M):
            ph1[i]=ph[i]
        t2 = Y[x] - (w1s.dot(ph1))
        t3 = - (ph1.dot(t2))
        dw = (eta*t3)
        de = dw/ TrainN
        WSN = WS - de
        w2s=np.transpose(WSN)
        """
        x1=phi.dot(WSN)
        x2=Y-x1
        x2=np.transpose(x2)
        x3=x2.dot(phi)
        x3=np.transpose(x3)
        de=-x3/TrainN #-------deltaEd------
        #print("de",de)
    
        dw=eta*de  #-----deltaW-------
        #print("dw",dw)
        WSN=WS-dw  #-----W(t+1)-------
        h2=0
        
        #print("i,WSN",i,WSN)
        """
        h2=0
        for j in range(0,TrainN):
            h=w2s.dot(phi[j])
            h2+=(Y[j]-h)*(Y[j]-h)
            EDNs=float(h2/2)
        #print("EDNs,EDs",EDNs,EDs)
        ER1=(2*EDNs)/TrainN
        ERMS2n=np.sqrt(ER1)#------Erms-----
        #print("ERMS2n,ERMS2i",ERMS2n,ERMS2i)
        temp=ERMS2i
        ERMS2i=ERMS2n
        WS=WSN
        #d=ERMS2n-temp
        #d=np.abs(d)
        #eta=(eta/2)
        """
        if(d<=0.001):
            break
        """
    print("Weights calculated from Stochastic Gradient Descent Method")
    print(WSN)
    w2s=np.transpose(WSN)
    

    #-------REgularized------------------------------------------------
    WSR=np.zeros((M,1),dtype='float')
    em=np.zeros((M,1),dtype='float')
    WSR = [[0],[0],[1],[0]]#,[0],[0],[0]]#,[0],[0]]
    w1rs=np.transpose(WSR)
    WSNR=WSR
    #---------Initial Error-------------
    h2=0
    #print("w1s,phip",w1s,phi[0])
    for p in range(0,TrainN):
        h=w1rs.dot(phi[p])
        h2+=(Y[p]-h)*(Y[p]-h)
    #print("h2n",h2)
    EDrs=float((h2/2))#-------/TrainN---------------
    #print("EDs",EDs)
    
    ER1r=(2*EDrs)/TrainN
    ERMS2ir=np.sqrt(ER1r)
    #print("initial error",ERMS2ir)
    #print("ERMS2",ERMS2i)
    #dEr1=ERMS2ir
    #Edr1=EDrs
    #------------Iteration--------
    eta=0.5
    for x in range(0,100):
        ph=phi[x]
        for i in range(0,M):
            ph1[i]=ph[i]
        x1=w1rs.dot(ph1)
        x2=Y[x]-x1
        x3=ph1.dot(x2)
        de=-x3   #-----deltaEd------
        em= [x * lamda for x in WS]
        #dew=WSR#----for REG,deltaEw----
        #deE=de+(lamda*dew)
        deE=de+em
        deE=deE/TrainN#-------deltaE for REG----
        dwr=eta*deE#----deltaW for REG----------
        WSNR=WSR-dwr#-----W(t+1)-------
        h2=0
        w2sr=np.transpose(WSNR)
        #print("i,WSN",i,WSN)
        for j in range(0,TrainN):
            h=w2sr.dot(phi[j])
            h2+=(Y[j]-h)*(Y[j]-h)
        EDNRs=float(h2/2)
        #print("EDNs,EDs",EDNs,EDs)
        ER1r=(2*EDNRs)/TrainN
        ERMS2nr=np.sqrt(ER1r)#------Erms-----
        #print("ERMS2n,ERMS2i",ERMS2n,ERMS2i)
        temp=ERMS2ir
        ERMS2ir=ERMS2nr
        WSR=WSNR
        d=ERMS2nr-temp
        d=np.abs(d)
        #eta=eta/2
        """
        if(d<=0.001):
            break
        """
    #print(temp,i-1)
    #print(ERMS2nr,i)
    print("Regularized Weights calculated from Stochastic Gradient Descent Method")
    print(WSNR)
    print("RMS Error for training data from Stochastic Gradient Descent Method")
    print(ERMS2n)
    print("Regularized RMS Error for training data from Stochastic Gradient Descent Method")
    print(ERMS2nr)
    w2sr=np.transpose(WSNR)
    
    #----------------Validation--------------------------------
    
    
    sigma1=np.zeros((46,46),dtype='float')
    for i in range(0,46):
        sigma1[i][i] = np.var(Xvalid[:,i])
        #print(sigma)
        #
        #M=int(M)
        
    mu1=np.zeros((M,46))
    #mu=1/46
    """
    for k in range(0,TrainN):
        for n in range(0,4):
            Z[k,n]=X[k,n]
            """
    #----------------------Generating Means----------------------
    for k in range(0,M):
        n1=random.randint(0,valid)
        mu1[k]=Xvalid[n1]
    #print("mu",len(mu[:,1]),len(mu[1,:]))
    #-----------------Calculating phi matrix----------------------
    phi1=np.zeros((valid,M),dtype='float')
    for i in range(0,valid):
        for j in range(0,M):
            if(j==0):
                phi1[i,j]=1
            else:
                t=(Xvalid[i]-mu1[j])
                #print("t",t)
                t2=np.transpose(t)
                #print("t2",t2)
                t3=sigma1.dot(t)
                #print("t3",t3)
                t4=t2.dot(t3)
                #print("t4",t4)
                #t4=t*t3
                t5=float(t4/2)
                phi1[i,j]=(np.exp(-t5))
    
    #-----------------Closed form weight-----------------
    
    h2=0
    #print("w1,phi[1]",w1,phi[1])
    for i in range(0,valid):
        h=w1.dot(phi1[i])
        h2+=(Yvalid[i]-h)*(Yvalid[i]-h)
        #print("h2",h2)
    EDW=float(h2/2) 
    #print("EDW",EDW)
    #--------------------Calculating ERMS-------------------
    #print("Valid: ",valid)
    ER1=(2*EDW)/valid
    ERMS=np.sqrt(ER1)
    print("RMS Error for validation data Closed Form Solution: ")
    print(ERMS)
    #-----------------Regularized Closed form weight------------
    
    #----------------Calculating ED Reg----------
    h2=0
    for i in range(0,valid):
        h=wr1.dot(phi1[i])
        h2+=(Yvalid[i]-h)*(Yvalid[i]-h)
        #print("h2",h2)
    EDWR=float(h2/2)
    #print("EDW",EDWR)
    
    #----------------ERMS Regularized----------------
    #Ereg=EDWR+lamda*EWR
    Ereg=EDWR
    Etemp=(2*Ereg)/valid
    ERMSreg=np.sqrt(Etemp)
    print("Regularized RMS Error for validation data Closed Form Solution : ")
    print(ERMSreg)
    
    #-----------------Stochastic Weight Validation-----------------
    
    h2=0
    #print("w1,phi[1]",w1,phi[1])
    for i in range(0,valid):
        h=w2s.dot(phi1[i])
        h2+=(Yvalid[i]-h)*(Yvalid[i]-h)
        #print("h2",h2)
    EDW=float(h2/2)
    #print("EDW",EDW)
    #--------------------Calculating ERMS-------------------
    ER1=(2*EDW)/valid
    ERMS=np.sqrt(ER1)
    print("RMS for validation data stochastic gradient descent: ")
    print(ERMS)

    #-----------------Regularized Stochastic Weights------------
    
    #----------------Calculating ED Reg----------
    h2=0
    for i in range(0,valid):
        h=w2sr.dot(phi1[i])
        h2+=(Yvalid[i]-h)*(Yvalid[i]-h)
        #print("h2",h2)
    EDWR=float(h2/2)
    #print("EDW",EDWR)
    
    #----------------ERMS Regularized----------------
    #Ereg=EDWR+lamda*EWR
    Ereg=EDWR
    Etemp=(2*Ereg)/valid
    ERMSreg=np.sqrt(Etemp)
    print("Regularized RMS for validation data stochastic gradient descent: ")
    print(ERMSreg)


    #------------------------Testing---------------------------------
    sigma2=np.zeros((46,46),dtype='float')
    for i in range(0,46):
        sigma2[i,i] = np.var(Xtest[:,i])
        #print(sigma)
        #
        #M=int(M)
        
    mu2=np.zeros((M,46))
    #mu=1/46
    """
    for k in range(0,TrainN):
        for n in range(0,4):
            Z[k,n]=X[k,n]
            """
    #----------------------Generating Means----------------------
    for k in range(0,M):
        n1=random.randint(0,Test)
        mu2[k]=Xtest[n1]
    #print("mu",len(mu[:,1]),len(mu[1,:]))
    #-----------------Calculating phi matrix----------------------
    phi2=np.zeros((Test,M),dtype='float')
    for i in range(0,Test):
        for j in range(0,M):
            if(j==0):
                phi2[i,j]=1
            else:
                t=(Xtest[i]-mu2[j])
                #print("t",t)
                t2=np.transpose(t)
                #print("t2",t2)
                t3=sigma2.dot(t)
                #print("t3",t3)
                t4=t2.dot(t3)
                #print("t4",t4)
                #t4=t*t3
                t5=float(t4/2)
                phi2[i,j]=(np.exp(-t5))

 #-----------------Closed form weight-----------------
    
    h2=0
    #print("w1,phi[1]",w1,phi[1])
    for i in range(0,Test):
        h=w1.dot(phi2[i])
        h2+=(Ytest[i]-h)*(Ytest[i]-h)
        #print("h2",h2)
    EDW=float(h2/2)
    #print("EDW",EDW)
    #--------------------Calculating ERMS-------------------
    #print("VAlid: ",Test)
    ER1=(2*EDW)/Test
    ERMS=np.sqrt(ER1)
    print("RMS for Testing data Closed Form Solution: ")
    print(ERMS)
    
    #-----------------Regularized Closed form weight------------
    
    #----------------Calculating ED Reg----------
    h2=0
    for i in range(0,Test):
        h=wr1.dot(phi2[i])
        h2+=(Ytest[i]-h)*(Ytest[i]-h)
        #print("h2",h2)
    EDWR=float(h2/2)
    #print("EDW",EDWR)
    
    #----------------ERMS Regularized----------------
    #Ereg=EDWR+lamda*EWR
    Ereg=EDWR
    Etemp=(2*Ereg)/Test
    ERMSreg=np.sqrt(Etemp)
    print("Regularized ERMS for Testing data Closed Form Solution: ")
    print(ERMSreg)
    
    #-----------------Stochastic Weight Testing-----------------
    
    h2=0
    #print("w1,phi[1]",w1,phi[1])
    for i in range(0,Test):
        h=w2s.dot(phi2[i])
        h2+=(Ytest[i]-h)*(Ytest[i]-h)
        #print("h2",h2)
    EDW=float(h2/2)
    #print("EDW",EDW)
    #--------------------Calculating ERMS-------------------
    ER1=(2*EDW)/Test
    ERMS=np.sqrt(ER1)
    print("ERMS for Testing data stochastic gradient descent: ")
    print(ERMS)

    #-----------------Regularized Stochastic Weights------------
    
    #----------------Calculating ED Reg----------
    h2=0
    for i in range(0,Test):
        h=w2sr.dot(phi2[i])
        h2+=(Ytest[i]-h)*(Ytest[i]-h)
        #print("h2",h2)
    EDWR=float(h2/2)
    #print("EDW",EDWR)
    
    #----------------ERMS Regularized----------------
    #Ereg=EDWR+lamda*EWR
    Ereg=EDWR
    Etemp=(2*Ereg)/Test
    ERMSreg=np.sqrt(Etemp)
    print("Regularized ERMS for Testing data stochastic gradient descent: ")
    print(ERMSreg)

    return

"""
def ValidTest(W1,W2,W3,W4,X,Y):
    
    sigma=np.zeros((46,46),dtype='float')
    for i in range(0,46):
        sigma[i][i] = np.var(X[:,i])
        #print(sigma)
        #
        #M=int(M)
        
    M=6
    mu=np.zeros((M,46))
    #mu=1/46
    
    for k in range(0,TrainN):
        for n in range(0,4):
            Z[k,n]=X[k,n]
    
    #----------------------Generating Means----------------------
    for k in range(0,M):
        n=random.randint(0,TrainN)
        mu[k]=X[n]
    #print("mu",len(mu[:,1]),len(mu[1,:]))
    #-----------------Calculating phi matrix----------------------
    phi=np.zeros((TrainN,M),dtype='float')
    for i in range(0,TrainN):
        for j in range(0,M):
            if(j==0):
                phi[i,j]=1
            else:
                t=(X[i]-mu[j])
                #print("t",t)
                t2=np.transpose(t)
                #print("t2",t2)
                t3=sigma.dot(t)
                #print("t3",t3)
                t4=t2.dot(t3)
                #print("t4",t4)
                #t4=t*t3
                t5=float(t4/2)
                phi[i,j]=(np.exp(-t5))    
    
    
    return

"""
print('---------------------For the data MQ2007----------------------------')
LetoR(X,Y,Xvalid,Yvalid,Xtest,Ytest)

#----------- CSV Data Training ------------

X1 = np.loadtxt(open("Querylevelnorm_X.csv","rb"),delimiter=",")
#print("CSV X: ",X1)

Y1 = np.loadtxt(open("Querylevelnorm_t.csv","rb"),delimiter=",")
#print("CSV Y: ",Y1)

#LetoR(X1,Y1)

for row in range(0,TrainN):
    for col in range(0,46):
        X2[row][col]=X1[row][col]
        
for row in range(0,TrainN):
    #print("Y1[row]",Y1[row])
    Y2[row]=Y1[row]
    
xv=0
for row in range(TrainN, valid):
    for col in range(0,46):
        X2Valid[xv][col]=X1[row][col]
    xv=xv+1

yv=0
for row in range(TrainN,valid):
    Y2Valid[yv]=Y1[row]
        
xt=0
for row in range(ValidN, TestN):
    for col in range(0,46):
        X2Test[xt][col]=X1[row][col]
    xt=xt+1

yt=0
for row in range(ValidN, TestN):
    Y2Test[yt]=Y1[row]
#print("CSV X: ",X2)
#print("CSV Y: ",Y2)
print("----------------------------For the Synthetic Data------------------")
LetoR(X2,Y2,X2Valid,Y2Valid,X2Test,Y2Test)
    

    
"""
---------------------PLOTS---------------------
import matplotlib.pyplot as plt

Mv=[[3],[4],[5],[6],[7],[9]]
ERMSv=[[0.310],[0.316],[.339],[.561],[.361],[.416]]
fig1=plt.figure()
ax1=fig1.add_subplot(111)
ax1.scatter(Mv,ERMSv)
plt.xlabel('M')
plt.ylabel('ERMS MQ2007 data')
plt.savefig('images/p1M.jpeg')

Mv=[[3],[4],[5],[6],[7],[9]]
ERMSv=[[0.310],[0.316],[.336],[.548],[.361],[.433]]
fig1=plt.figure()
ax1=fig1.add_subplot(111)
ax1.scatter(Mv,ERMSv)
plt.xlabel('M')
plt.ylabel('Regularized ERMS MQ2007 data')
plt.savefig('images/p2M.jpeg')

Mv=[[3],[4],[5],[6],[7],[9]]
ERMSv=[[0.293],[0.293],[.299],[.669],[.718],[.847]]
fig1=plt.figure()
ax1=fig1.add_subplot(111)
ax1.scatter(Mv,ERMSv)
plt.xlabel('M')
plt.ylabel('ERMS MQ2007 data Stochastic Gradient Descent')
plt.savefig('images/p3M.jpeg')

Mv=[[3],[4],[5],[6],[7],[9]]
ERMSv=[[0.293],[0.293],[.299],[.669],[.718],[.847]]
fig1=plt.figure()
ax1=fig1.add_subplot(111)
ax1.scatter(Mv,ERMSv)
plt.xlabel('M')
plt.ylabel('Regularized ERMS MQ2007 data Stochastic Gradient Descent')
plt.savefig('images/p4M.jpeg')

Mv=[[3],[4],[5],[6],[7],[9]]
ERMSv=[[0.323],[0.323],[.360],[.286],[.389],[.414]]
fig1=plt.figure()
ax1=fig1.add_subplot(111)
ax1.scatter(Mv,ERMSv)
plt.xlabel('M')
plt.ylabel('ERMS synthetic data')
plt.savefig('images/p5M.jpeg')

Mv=[[3],[4],[5],[6],[7],[9]]
ERMSv=[[0.323],[0.323],[.361],[.282],[.389],[.409]]
fig1=plt.figure()
ax1=fig1.add_subplot(111)
ax1.scatter(Mv,ERMSv)
plt.xlabel('M')
plt.ylabel('Regularized ERMS synthetic data')
plt.savefig('images/p6M.jpeg')

Mv=[[3],[4],[5],[6],[7],[9]]
ERMSv=[[0.307],[0.307],[.483],[.997],[.998],[.996]]
fig1=plt.figure()
ax1=fig1.add_subplot(111)
ax1.scatter(Mv,ERMSv)
plt.xlabel('M')
plt.ylabel('ERMS synthetic data using Stochastic Gradient Descent')
plt.savefig('images/p7M.jpeg')

Mv=[[3],[4],[5],[6],[7],[9]]
ERMSv=[[0.307],[0.307],[.483],[.997],[.998],[.996]]
fig1=plt.figure()
ax1=fig1.add_subplot(111)
ax1.scatter(Mv,ERMSv)
plt.xlabel('M')
plt.ylabel('Regularized ERMS synthetic data using Stochastic Gradient Descent')
plt.savefig('images/p8M.jpeg')

Ev=[[0.3],[0.4],[0.5],[0.6],[0.9]]
ERMSv=[[0.304],[0.787],[.316],[.595],[.351]]
fig1=plt.figure()
ax1=fig1.add_subplot(111)
ax1.scatter(Ev,ERMSv)
plt.xlabel('eta')
plt.ylabel('ERMS MQ2007 data')
plt.savefig('images/p1E.jpeg')

Ev=[[0.3],[0.4],[0.5],[0.6],[0.9]]
ERMSv=[[0.303],[0.781],[.316],[.594],[.351]]
fig1=plt.figure()
ax1=fig1.add_subplot(111)
ax1.scatter(Ev,ERMSv)
plt.xlabel('eta')
plt.ylabel('Regularized ERMS MQ2007 data')
plt.savefig('images/p2E.jpeg')

Ev=[[0.3],[0.4],[0.5],[0.6],[0.9]]
ERMSv=[[0.319],[0.733],[.293],[.456],[.852]]
fig1=plt.figure()
ax1=fig1.add_subplot(111)
ax1.scatter(Ev,ERMSv)
plt.xlabel('eta')
plt.ylabel('ERMS MQ2007 data Stochastic Gradient Descent')
plt.savefig('images/p3E.jpeg')

Ev=[[0.3],[0.4],[0.5],[0.6],[0.9]]
ERMSv=[[0.518],[0.732],[.293],[.455],[.852]]
fig1=plt.figure()
ax1=fig1.add_subplot(111)
ax1.scatter(Ev,ERMSv)
plt.xlabel('eta')
plt.ylabel('Regularized ERMS MQ2007 data Stochastic Gradient Descent')
plt.savefig('images/p4E.jpeg')

Ev=[[0.3],[0.4],[0.5],[0.6],[0.9]]
ERMSv=[[0.342],[0.291],[.323],[.359],[.305]]
fig1=plt.figure()
ax1=fig1.add_subplot(111)
ax1.scatter(Ev,ERMSv)
plt.xlabel('eta')
plt.ylabel('ERMS synthetic data')
plt.savefig('images/p5E.jpeg')

Ev=[[0.3],[0.4],[0.5],[0.6],[0.9]]
ERMSv=[[0.342],[0.291],[.323],[.359],[.305]]
fig1=plt.figure()
ax1=fig1.add_subplot(111)
ax1.scatter(Ev,ERMSv)
plt.xlabel('eta')
plt.ylabel('Regularized ERMS synthetic data')
plt.savefig('images/p6E.jpeg')

Ev=[[0.3],[0.4],[0.5],[0.6],[0.9]]
ERMSv=[[0.899],[0.998],[.807],[.960],[.997]]
fig1=plt.figure()
ax1=fig1.add_subplot(111)
ax1.scatter(Ev,ERMSv)
plt.xlabel('eta')
plt.ylabel('ERMS synthetic data using Stochastic Gradient Descent')
plt.savefig('images/p7E.jpeg')

Ev=[[0.3],[0.4],[0.5],[0.6],[0.9]]
ERMSv=[[0.899],[0.998],[.807],[.960],[.996]]
fig1=plt.figure()
ax1=fig1.add_subplot(111)
ax1.scatter(Ev,ERMSv)
plt.xlabel('eta')
plt.ylabel('Regularized ERMS synthetic data using Stochastic Gradient Descent')
plt.savefig('images/p8E.jpeg')
"""




