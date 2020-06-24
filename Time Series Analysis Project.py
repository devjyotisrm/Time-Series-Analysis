#!/usr/bin/env python
# coding: utf-8

# In[122]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.linear_model import LinearRegression


# In[123]:


data = pd.read_excel('C:/Users/DELL/Desktop/Superstore.xlsx')


# In[124]:


data.head()


# In[125]:


furniture = data.loc[data['Category']== 'Furniture']


# In[126]:


furniture.head()


# In[127]:


furniture['Order Date'].min()


# In[128]:


furniture['Order Date'].max()


# In[129]:


cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']
furniture.drop(cols, axis=1, inplace=True)


# In[130]:


furniture = furniture.sort_values('Order Date')


# In[131]:


furniture.isnull().sum()


# In[132]:


furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()


# In[133]:


furniture


# In[134]:


furniture = furniture.set_index('Order Date')


# In[135]:


furniture.index


# In[136]:


furniture


# In[137]:


y = furniture['Sales'].resample('MS').mean()


# In[138]:


y


# In[139]:


y.plot(figsize=(15, 6))
plt.show()


# In[140]:


# Rolling Statistics 
rollmean = y.rolling(window =12).mean()
rollstd = y.rolling(window =12).std()
print(rollmean,rollstd)


# In[141]:


# Plotting rolling statistics
plt.plot(y,color='blue',label='original')
plt.plot(rollmean,color='red',label='rolling mean')
plt.plot(rollstd,color='black',label='rolling std')
plt.title('Rolling mean and Rolling Standard Deviation')
plt.show()


# In[ ]:





# In[142]:


from pylab import rcParams
rcParams['figure.figsize']= 18,8
decomposition = sm.tsa.seasonal_decompose(y,model='additive')
fig = decomposition.plot()
plt.show()


# In[143]:


plot_pacf(y,lags=20)


# In[144]:


y.shape


# In[145]:


y = y.to_frame()


# In[146]:


y


# In[147]:


y['Sales_TwelfthLag'] = y['Sales'].shift(12,axis=0)


# In[148]:


y.head(14)


# In[149]:


y.dropna(inplace=True)


# In[150]:


y.head()


# In[154]:


Y = y.Sales.values
X = y.Sales_TwelfthLag.values


# In[155]:


train_size = int(len(X)*0.80)


# In[160]:


X_train,X_test = X[0:train_size],X[train_size:len(X)]
Y_train,Y_test = Y[0:train_size],Y[train_size:len(X)]


# In[161]:


X_train = X_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)


# In[162]:


regressor = LinearRegression()
regressor.fit(X_train,Y_train)


# In[163]:


regressor.coef_


# In[164]:


regressor.intercept_


# In[165]:


Y_pred = regressor.predict(X_test)


# In[166]:


plt.plot(Y_test[-10:],label = 'Actual Values')
plt.plot(Y_pred[-10:],label = 'Predicted Values')
plt.legend()
plt.show()


# In[167]:


model = ARIMA(Y_train,order=(1,0,0))


# In[168]:


model_fit = model.fit()


# In[169]:


print(model_fit.summary())


# In[ ]:




