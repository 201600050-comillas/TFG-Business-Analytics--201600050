#!/usr/bin/env python
# coding: utf-8

# In[35]:


#CARGA DE LIBRERÍAS
import pandas as pd
import matplotlib.pyplot as plt


# In[36]:


#CARGA DE DATOS DJSI Europe y S&P Europe BMI
dfES = pd.read_excel (r'/Users/rocioartiaga/Desktop/indices/djsi-returns.xlsx')


# In[37]:


#DETERMINACIÓN DE LOS EJES
dfES.set_axis(['Date', 'DJ Sustainability Europe', 'S&P Europe BMI'], 
                    axis='columns', inplace=True)


# In[38]:


#CAMBIO DE FORMATO DE LA VARIABLE FECHA A FORMATO DATETIME
dfES['Date']= pd.to_datetime(dfES['Date'])


# In[39]:


#ESTABLECER EL ÍNDICE
dfES.set_index('Date',inplace=True)


# In[40]:


#GRÁFICO DJSI Europe y S&P Europe BMI
ax=dfES['S&P Europe BMI'].plot(title='EVOLUCIÓN DEL RENDIMIENTO DE ÍNDICES EUROPEOS',figsize=(20, 12),c='blue',legend='S&P Europe BMI')
dfES['DJ Sustainability Europe'].plot(ax=ax,c="grey",legend='DJ Sustainability Europe')


# In[52]:


#CARGA DE DATOS EURO STOXX Sustainability y Euro Stoxx
dfES = pd.read_excel (r'/Users/rocioartiaga/Desktop/indices/return-EuroStoxx.xlsx')


# In[53]:


#ESTABLECER EJES
dfES.set_axis(['Date', 'Euro Stoxx Sustainability', 'Euro Stoxx'], 
                    axis='columns', inplace=True)


# In[54]:


#CAMBIO DE FORMATO
dfES['Date']= pd.to_datetime(dfES['Date'])


# In[55]:


#ESTABLECER ÍNDICE
dfES.set_index('Date',inplace=True)


# In[57]:


#GRÁFICO
ax=dfES['Euro Stoxx Sustainability'].plot(title='EVOLUCIÓN DEL RENDIMIENTO DE ÍNDICES EUROPEOS',figsize=(20, 12),c='grey',legend='Euro Stoxx Sustainability')
dfES['Euro Stoxx'].plot(ax=ax,c="blue",legend='Euro Stoxx')
ax.legend()


# In[58]:


#CARGA DATOS FTSE 4GOOD IBEX e IBEX35
dfIBEX = pd.read_excel (r'/Users/rocioartiaga/Desktop/indices/ibex/ibex-juntos.xlsx')


# In[59]:


#EJES
dfIBEX.set_axis(['Date', 'FTSE4GOOD_IBEX', 'IBEX35'], 
                    axis='columns', inplace=True)


# In[60]:


#ÍNDICE
dfIBEX.set_index('Date',inplace=True)


# In[61]:


#GRÁFICO
ax=dfIBEX['FTSE4GOOD_IBEX'].plot(title='RENDIMIENTO ÍNDICES ESPAÑOLES',figsize=(20, 12),c='grey',legend='FTSE 4 Good IBEX')
dfIBEX['IBEX35'].plot(ax=ax,c="blue",legend='IBEX35')
ax.legend()


# In[ ]:




