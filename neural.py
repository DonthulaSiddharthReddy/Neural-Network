import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("new_final.csv")

df=df.set_index('ID')

data=np.array(df)
p,q=np.shape(data)

def normalize(data):
  m,n=np.shape(data)
  for j in range(n-1):
    max1=np.max(data[:,j])
    min1=np.min(data[:,j])
    for i in range(m):
      data[i,j]=(data[i,j]-min1)/(max1-min1)

normalize(data)

def convert(Y):#converting g to 1 and h to zero
  y=np.zeros((1,np.size(Y)))
  for i in range(np.size(Y)):
    if Y[i,0]=='g':
      y[0,i]=1
    else:
      y[0,i]=0
  return y

def deconvert(Y):#converting 1 to g and 0 to h
	y=np.zeros((1,np.size(Y))).astype('str')
	for i in range(np.size(Y)):
		if Y[0,i]==1:
			y[0,i]="g"
		else:
			y[0,i]="h"
	return y

y=data[:,-1].reshape(-1,1)
print(y)

X=np.delete(data,-1,1)#input set

# to add bais term in every layer
def bais(x):
	ones=np.array([(1)])
	#for i in range(99):
	#ones=np.vstack((ones,np.array([(1)])))
	u=np.hstack((ones,x))
	return u

# to intials theta randomly
def weight(layer1,layer2):
  w=np.random.rand(layer1+1,layer2)
  #w=np.zeros((m,l))
  return w

def sigmoid(c):
	return 1/(1+(np.exp(-c)))

#------parameters--------#

alpha=0.01
e=1e-5
ilter=range(50)
lamb=0

#-------inilizing weigths---------#
m,n=X.shape

theta_1=weight(n,5)# hidden layer with 5 nodes
theta_2=weight(5,1)# output



def layer(x,theta):
  n=sigmoid(bais(x)@theta)
  return n

def Forward_propogation(X):
  n1=X

  n2=layer(n1,theta_1)

  n3=layer(n2,theta_2)

  return n1,n2,n3

def Back_propogation(X):
  
  n1,n2,n3=Forward_propogation(X)
  
  delta_3=n3-y[i,0]

  delta_2=((theta_2[1:,:])@delta_3)-(n2*(1-n2)) #2th layer error is backpropagated to 2th layer with its weigths

  d1=(bais(n1).reshape(-1,1))@(delta_2.reshape(1,n2.shape[0]))

  d2=(bais(n2).reshape(-1,1))@(delta_3.reshape(1,n3.shape[0]))

  return d1,d2,n3

def gradeint_descent(X):
  m,n=X.shape
  K=np.array([(0)])
  L1=0
  L2=0
  for i in range(m):
    Z=X[i,0:]
    d1,d2,n3=Back_propogation(Z)
    L1=L1+d1
    L2=L2+d2
    K=np.vstack((K,n3))
    
  K=K[1:,:]# removing sample element

  #---------differentiation term-----------#

  D1=(1/m)*(L1[1:,:]+((lamb)*theta_1[1:,:]))
  
  D2=(1/m)*(L2[1:,:]+((lamb)*theta_2[1:,:]))
  
  D10=(1/m)*(L1[0,:])
  
  D20=(1/m)*(L2[0,:])
 

  #----------gradient descent--------------#

  theta_1[1:,:]=theta_1[1:,:]-((alpha/m)*D1)
  
  theta_2[1:,:]=theta_2[1:,:]-((alpha/m)*D2)
  
  theta_1[0,:]=theta_1[0,:]-((alpha/m)*D10)
  
  theta_2[0,:]=theta_2[0,:]-((alpha/m)*D20)
  
  return K

def cost_function(X):
  K=gradeint_descent(X)
  J=(-(1/m))*np.sum((y*np.log(K+e))+((1-y)*np.log(1-K+e)))
  return J,K

k=np.array([(0)])
for i in ilter:
  print(i)
  J,K=cost_function(X)
  k=np.vstack((k,np.array(J)))
  
print(J)
print(K)
print('###################')
print(k)


plt.plot(ilter,k[1:,:], label='Decision Boundary')
plt.show()


