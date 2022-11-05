#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize


# In[2]:


# importing the data from the given csv file

df = pd.read_csv('04_cricket_1999to2011.csv')


# In[3]:


# eliminating the  second innings data
df = df[df['Innings']==1]

# taking the match number of matches in which the first innings was not completely played due to rain or someother reasons
Match_Number = df[(df['Wickets.in.Hand']!=0)&(df['Over']!=50)&(df['Total.Runs']==df['Innings.Total.Runs'])]['Match'].values.tolist()


# In[4]:


# removing the data of the matches which was obtained in the previous step
for i in Match_Number:
    df.drop(df.index[df['Match'] == i], inplace=True)

# from the data given we require only the over, runs scored and wickets in hand data
df = df[["Over","Runs.Remaining","Wickets.in.Hand"]]

# converting the overs used to overs remaining values
df['Over']=50-df['Over']


# In[5]:


# defining a function to load data according to the wickets remaining in hand which returns the loaded data
def load_data():
    Data=[]
    for i in range(0,11):
        A_ = df[df['Wickets.in.Hand'] == i].values.tolist()
        Data.append(A_)
    return Data


# In[6]:


# defining a function which takes the loaded data as input and seperates themm into overs and runs and returns them
def get_data(Data):
    A=[]
    B=[]
    for i in range(1,len(Data)):
            for j in range(0,len(Data[i])):
                A.append(Data[i][j][0])
                B.append(Data[i][j][1])
    u = np.array(A)
    v = np.array(B)
    return u,v


# In[7]:


# defining a function which takes the parameters as input and gives the prediction as output
def pred(Z_0,L,u):
    return (Z_0*(1-math.exp(-(L*u)/(Z_0))))


# In[8]:


# defining a function which gives a list that contains the number of data for each wicket
def get_length():
    Data = load_data()
    l=[]
    l.append(0)
    sum=0
    for i in range(1,len(Data)):
        sum += len(Data[i])
        l.append(sum)
        
    return l


# In[9]:


# defining the square error loss function which has to be optimized 
def loss(a,u,v):
    sum = 0
    l = get_length()
    for j in range(1,11):
        for i in range(l[j-1],l[j]):
            Z = pred(a[j],a[0],u[i])
            sum += (Z-v[i])**2
    return sum/len(u)


# In[10]:


# Z which is the initial values for the parameters for optimizing the loss function
Z=[6,18,48,65,88,123,143,151,176,183,289]


# In[11]:


# getting the data
Data = load_data()
u,v = get_data(Data)


# In[12]:


# using scipy optimization to minimize the loss function
min = minimize(loss,Z,args=(u,v),method='BFGS')


# In[13]:


min


# In[14]:


Z=min.x # from the minimize function we take the parameters Z_0 and L


# In[15]:


# defining function which gives the total loss without normalizing
def Total_loss(a,u,v):
    sum = 0
    l = get_length()
    for j in range(1,11):
        for i in range(l[j-1],l[j]):
            Z = pred(a[j],a[0],u[i])
            sum += (Z-v[i])**2
    return sum


# In[16]:


Total_loss(Z,u,v)


# In[17]:


# defining a function which takes the parameters as input and gives out the data for plotting
def get_plot_data(Z):    
    fn = np.arange(510)
    fn = fn.reshape(10,51)
    for i in range(1,11):
        for j in range(0,51):
            fn[i-1][j] = pred(Z[i],Z[0],j)
    return fn


# In[19]:


# plotting of Average Runs Obtainable vs Overs Remaining

Overs = np.arange(51)
fn = get_plot_data(Z)
A = ['1','2','3','4','5','6','7','8','9','10']
plt.figure(figsize=(15, 10))
plt.axis([0, 50, 0, 300])
for i in range(0,10):
    plt.plot(Overs,fn[i], label = A[i])
plt.xlabel('Overs Remaining')
plt.ylabel('Average Runs Obtainable')
plt.title("Average Runs Obtainable vs Overs Remaining")
plt.legend(loc='upper left')
plt.show()

