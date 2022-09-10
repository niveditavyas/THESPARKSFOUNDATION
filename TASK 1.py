#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[3]:


df = pd.read_csv('taskone.csv')
df.head()


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
df.plot(x='Hours',y='Scores',style='o')
plt.title('Hours vs Score')
plt.xlabel('Hours Studied')
plt.ylabel("Percentage Scores")
plt.show()


# In[5]:


X = df.iloc[:,:-1].values
y = df.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


# In[7]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)


# In[8]:


#regression line
line = reg.coef_*X+reg.intercept_

plt.scatter(X,y)
plt.plot(X,line)
plt.show()


# In[12]:


y_pred = reg.predict(X_test)
df_com = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_com


# In[13]:


from sklearn import metrics 
error = metrics.mean_absolute_error(y_test, y_pred)
error

