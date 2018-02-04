from array import *
import numpy as np
import csv
from math import * 
import scipy.stats 
print "UBitName = manishat"

print "personNumber = 50207628"

ist1=[]
ist2=[]
ist3=[]
ist4=[]

with open('data.csv') as csvfile:
  wb = csv.reader(csvfile, delimiter=',')
  for row in wb:
      c=row[2] 
      r=row[3]
      a=row[4]
      t=row[5] 
      ist1.append(c)
      ist2.append(r)
      ist3.append(a)
      ist4.append(t)      

for i in range (1,51,1):
    ist1[i-1]=ist1[i]
    ist2[i-1]=ist2[i]
    ist3[i-1]=ist3[i]
    ist4[i-1]=ist4[i]
    


del ist1[49:]
del ist2[49:]
del ist3[49:]
del ist4[49:]

list1=array('d')
list2=array('d')
list3=array('i')
list4=array('i') 

list1=np.asarray(ist1,dtype=np.float64 )
list2=np.asarray(ist2,dtype=np.float64)
list3=np.asarray(ist3,dtype=np.int64)
list4=np.asarray(ist4,dtype=np.int64)
 

"""sheet1 = xlsheet['C2':'C50']
sheet2 = xlsheet['D2':'D50']
sheet3 = xlsheet['E2':'E50']
sheet4 = xlsheet['F2':'F50']
for i in sheet1:
	for obj in i:
		list1.append(obj.value)
for i in sheet2:
	for obj in i:
		list2.append(obj.value)
for i in sheet3:
	for obj in i:
		list3.append(obj.value)
for i in sheet4:
	for obj in i:
		list4.append(obj.value)"""
		
mu1 = np.mean(list1)
print "mu1=",round(mu1,3)
var1 = np.var(list1)
print "var1=",round(var1,3)
sigma1 = np.std(list1)
print "sigma1=",round(sigma1,3)
mu2 = np.mean(list2)
print "mu2=",round(mu2,3)
var2 = np.var(list2)
print "var2=",round(var2,3)
sigma2 = np.std(list2)
print "sigma2=",round(sigma2,3)
mu3 = np.mean(list3)
print "mu3=",round(mu3,3)
var3 = np.var(list3)
print "var3=",round(var3,3)
sigma3 = np.std(list3)
print "sigma3=",round(sigma3,3)
mu4 = np.mean(list4)
print "mu4=",round(mu4,3)
var4 = np.var(list4)
print "var4=",round(var4,3)
sigma4 = np.std(list4)
print "sigma4=",round(sigma4,3)
#End of problem 1


cov=(4,4)
cor=(4,4)
np.zeros(cov)
np.zeros(cor)

cova = np.cov(np.vstack((list1,list2,list3,list4)))
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
print "CovarianceMat = "
print np.around(cova,decimals = 3)
"most correlated pair = (1,2)"
"least correlated pair = (3,4)"
#Cov = np.matrix([[mat1.item(0), mat1.item(1),mat2.item(1), mat3.item(1)], [mat1.item(1), mat1.item(3), mat4.item(1),mat5.item(1)],[mat2.item(1),mat4.item(1),mat6.item(0),mat6.item(1)], [mat3.item(1), mat5.item(1), mat6.item(1),mat3.item(3)]])
#print np.around(cova,decimals = 2)
#print "CovarianceMat = " ,cova

corr = np.corrcoef(np.vstack((list1,list2,list3,list4)))
print "CorrelationMat = " 
print np.around(corr,decimals = 3)

#End of problem 2


p1=0
p2=0
p3=0
p4=0
for i in list1:
	p1 += log(scipy.stats.norm(mu1,sigma1).pdf(i))
for i in list2:
	p2 += log(scipy.stats.norm(mu2,sigma2).pdf(i))
for i in list3:
	p3 += log(scipy.stats.norm(mu3,sigma3).pdf(i))
for i in list4:
	p4 += log(scipy.stats.norm(mu4,sigma4).pdf(i))
print "LogLikelihood =",round(p1+p2+p3+p4,3)
#End of problem 3



#BN Graph  
print "BNGraph="
print np.matrix(([0,1,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]))

 
 
 
 
#Calculating probability assuming research_overhead(X2) is dependent on CSScore(X1) and Tuition(X4)
#Matrix A
A = np.zeros((3,3))

dummy=np.zeros(49)


for i in range(0,49,1):
    dummy[i]=1


A[0][0]=np.dot(dummy,dummy)
A[0][1]=np.dot(dummy,list1)
A[0][2]=np.dot(dummy,list4)
A[1][0]=A[0][1]
A[1][1]=np.dot(list1,list1)
A[1][2]=np.dot(list1,list4)
A[2][0]=A[0][2]
A[2][1]=A[1][2]
A[2][2]=np.dot(list4,list4)

#Matrix Y

Y = np.zeros((3,1))
Y[0][0]=np.dot(list2,dummy)
Y[1][0]=np.dot(list2,list1)
Y[2][0]=np.dot(list2,list4)

#Matrix Beta

B=np.zeros((3,1))
B=np.dot(np.linalg.inv(A),Y)


#variance
variance = 0
for k in range(0,49,1):
	variance +=(B[0][0]*dummy[k]+B[1][0]*list1[k]+B[2][0]*list4[k]-list2[k])*(B[0][0]*dummy[k]+B[1][0]*list1[k]+B[2][0]*list4[k]-list2[k])
variance = variance/49
#loglikelihood
loglikelihood = 0
for k in range(0,49,1):
    loglikelihood+= (-1/(2*variance))*(B[0][0]*dummy[k]+B[1][0]*list1[k]+B[2][0]*list4[k]-list2[k])*(B[0][0]*dummy[k]+B[1][0]*list1[k]+B[2][0]*list4[k]-list2[k])
t = -(log(2*3.14*variance))*0.5*49
loglikelihood += t
loglikelihood += p1+p3+p4
print "BNLikelihood = ",loglikelihood
#End of problem 4








