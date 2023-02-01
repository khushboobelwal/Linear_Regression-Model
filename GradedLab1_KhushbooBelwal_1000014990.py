#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import datasets,linear_model,metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 


# In[11]:


df=pd.read_csv("C:/Users/DITU/Downloads/Salary_Data.csv")


# In[13]:


df.head()


# In[14]:


df.describe()


# In[15]:



df=df.dropna()


# In[16]:


y=df["YearsExperience"]
x=df["Salary"]


# In[17]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4)


# In[18]:


reg=LinearRegression()


# In[20]:


reg.fit(x_train.values.reshape(-1,1),y_train)
y_pred=reg.predict(x_test.values.reshape(-1,1))
print(reg.score(y_pred.reshape(-1,1),y_test))


# In[ ]:




