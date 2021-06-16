#!/usr/bin/env python
# coding: utf-8

# In[1]:


#CARGA DE LIBRERIAS
import pandas as pd
from sklearn.linear_model import LinearRegression 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from scipy import stats
from statsmodels.regression.rolling import RollingOLS
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import normaltest
from scipy.stats import chisquare


# In[2]:


#CARGA DE DATOS DOW JONES SUSTAINABILITY INDEX
dfEU = pd.read_excel (r'/Users/rocioartiaga/Desktop/indices/dow-diarios3.xlsx')


# In[3]:


#EJES EN BASE DE DATOS
dfEU.set_axis(['Date', 'Utilities', 'HC','DJSI', 'SP','EN'], 
                    axis='columns', inplace=True)


# In[4]:


#CÁLCULO DEL RENDIMIENTO DIARIO
dfEU["Utilities"]= dfEU["Utilities"].pct_change(-1)
dfEU["HC"]= dfEU["HC"].pct_change(-1)
dfEU["DJSI"]= dfEU["DJSI"].pct_change(-1)
dfEU["SP"]= dfEU["SP"].pct_change(-1)
dfEU['EN']=dfEU['EN'].pct_change(-1)


# In[5]:


#ELIMINACIÓN DE LOS VALORES NA (FALTANTES)
dfEU.dropna()


# In[6]:


#CÁLCULO DE LA PRIMA ESG
dfEU['primaESG']= dfEU['DJSI']- dfEU['SP']


# In[7]:


#ORDENAR POR FECHA ASCENDENTE
dfEU=dfEU.sort_values(by='Date')


# In[8]:


#TRATAMIENTO DE DATOS SUSTITUIR VALORES INFINITOS Y FALTANTES
dfEU.replace([np.inf, -np.inf], np.nan)

dfEU.dropna(inplace=True)


# In[9]:


#ESTRATIFICACIÓN DE LA MUESTRA POR PERÍODOS
#PERÍODO 1
start_date1 = "2010-8-04"
end_date1 = "2011-12-31"

after_start_date1 = dfEU["Date"] >= start_date1
before_end_date1 = dfEU["Date"] <= end_date1
between_two_dates1 = after_start_date1 & before_end_date1
filtered_dates1 = dfEU.loc[between_two_dates1]
subset1= filtered_dates1


# In[10]:


#PERÍODO 2
start_date2 = "2012-1-1"
end_date2 = "2013-12-31"

after_start_date2 = dfEU["Date"] >= start_date2
before_end_date2 = dfEU["Date"] <= end_date2
between_two_dates2 = after_start_date2 & before_end_date2
filtered_dates2 = dfEU.loc[between_two_dates2]
subset2= filtered_dates2


# In[11]:


#PERÍODO 3
start_date3 = "2014-1-1"
end_date3 = "2015-12-31"

after_start_date3 = dfEU["Date"] >= start_date3
before_end_date3 = dfEU["Date"] <= end_date3
between_two_dates3 = after_start_date3 & before_end_date3
filtered_dates3 = dfEU.loc[between_two_dates3]
subset3= filtered_dates3


# In[12]:


#PERÍODO 4
start_date4 = "2016-1-1"
end_date4 = "2017-12-31"

after_start_date4 = dfEU["Date"] >= start_date4
before_end_date4 = dfEU["Date"] <= end_date4
between_two_dates4 = after_start_date4 & before_end_date4
filtered_dates4 = dfEU.loc[between_two_dates4]
subset4= filtered_dates4


# In[13]:


#PERÍODO 5
start_date5 = "2018-1-1"
end_date5 = "2019-12-31"

after_start_date5 = dfEU["Date"] >= start_date5
before_end_date5 = dfEU["Date"] <= end_date5
between_two_dates5 = after_start_date5 & before_end_date5
filtered_dates5 = dfEU.loc[between_two_dates5]
subset5= filtered_dates5


# In[14]:


#PERÍODO 6
start_date6 = "2020-1-1"
end_date6 = "2021-3-8"

after_start_date6 = dfEU["Date"] >= start_date6
before_end_date6 = dfEU["Date"] <= end_date6
between_two_dates6 = after_start_date6 & before_end_date6
filtered_dates6 = dfEU.loc[between_two_dates6]
subset6= filtered_dates6


# In[15]:


#PERÍODO 7: PERÍODO DESDE 2012 PARA CALCULAR POSTERIORMENTE SHARPE RATIO
start_date7 = "2012-1-1"
end_date7 = "2021-3-8"

after_start_date7 = dfEU["Date"] >= start_date7
before_end_date7 = dfEU["Date"] <= end_date7
between_two_dates7 = after_start_date7 & before_end_date7
filtered_dates7 = dfEU.loc[between_two_dates7]
subset7= filtered_dates7


# In[16]:


#ESTABLECER EL ÍNDICE 
dfEU=dfEU.set_index('Date')


# In[17]:


#CÁLCULO CON ROLLING REGRESSION PRIMA ESG 

items= dfEU[['Utilities', 'HC','EN']]

for i in items:

    X = dfEU[['SP', 'primaESG']]
    y = items[i]
    
    exog = sm.add_constant(X)
    rols = RollingOLS(endog=y, exog=exog,window=253,)
    modelo = rols.fit()
    params=modelo.params
    fig = modelo.plot_recursive_coefficient(variables=['primaESG'], figsize=(14,6)) #Gráfico del coeficiente asignado a la prima ESG


# In[18]:


dfEU['primaESG'].rolling(253).corr(dfEU['HC']).plot (title='CORRELACIÓN PRIMA ESG EN EL SECTOR DE LA SALUD', figsize=(20,12), color='blue')


# In[19]:


dfEU['primaESG'].rolling(253).corr(dfEU['Utilities']).plot (title='CORRELACIÓN PRIMA ESG EN EL SECTOR SERVICIOS', figsize=(20,12), color='blue')


# In[20]:


dfEU['primaESG'].rolling(253).corr(dfEU['EN']).plot (title='CORRELACIÓN PRIMA ESG EN EL SECTOR ENERGÉTICO', figsize=(20,12), color='blue')


# In[21]:


#DISTRIBUCIÓN DE LAS VARIABLES
fig, axes = plt.subplots(2, 3, figsize=(15, 5), sharey=True)

sns.distplot(dfEU['SP'] , color="blue", ax=axes[0,0], axlabel='S&P Europe BMI')
sns.distplot(dfEU['DJSI'] , color="blue", ax=axes[0,1], axlabel='DJSI Europe')
sns.distplot(dfEU['primaESG'] , color="blue", ax=axes[1,1], axlabel= 'PrimaESG')
sns.distplot(dfEU['Utilities'] , color="blue", ax=axes[1,2], axlabel='Utilities')
sns.distplot(dfEU['EN'] , color="blue", ax=axes[1,0], axlabel= 'Energy')
sns.distplot(dfEU['HC'] , color="blue", ax=axes[0,2], axlabel='Health Care')
plt.subplots_adjust(top=1.5)


# In[22]:


#DESCRIPCIÓN DE LAS VARIABLES
dfEU.describe()


# In[23]:


#DEFINIR FUNCIÓN DE ROLLING SHARPE RATIO
def my_rolling_sharpe(y):
    return np.sqrt(253) * (y.mean() / y.std()) 


# In[24]:


subset7=subset7.set_index('Date')


# In[25]:


#CÁLCULO DE SHARPE RATIO
subset7['DJSI EUROPE'] = subset7['DJSI'].rolling('253d').apply(my_rolling_sharpe)
subset7['S&P EUROPE BMI'] = subset7['SP'].rolling('253d').apply(my_rolling_sharpe)


# In[26]:


#GRÁFICA DE SHARPE RATIO
subset7.plot(title= 'SHARPE RATIO DJSI EUROPE AND S&P EUROPE BMI',y=['DJSI EUROPE', 'S&P EUROPE BMI'], figsize=(20,12), color=('grey','blue'))


# In[27]:


plt.rcParams['image.cmap'] = "bwr"
#plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')


# In[28]:


#ESTIMACIÓN DEL MODELO DE REGRESIÓN LINEAL PARA EL PERÍODO ENTERO

#SE ESTABLECEN LOS SECTORES A ANALIZAR 
items= dfEU[['Utilities', 'HC','EN']] 

#se emplea un loop para simplificar el código y ser más eficiente
for i in items:

    X = dfEU[['SP', 'primaESG']] #VARIABLE X
    y = items[i] #VARIABLE Y 
    
    X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8, #conjunto de entrenamiento
                                        random_state = 1234,
                                        shuffle      = True
                                    )
    X_train = sm.add_constant(X_train, prepend=True)
    modelo = sm.OLS(endog=y_train, exog=X_train,)
    #ajusta el modelo
    modelo = modelo.fit()
    #imprime el modelo y los parámetros
    print("El modelo para el sector", [i], modelo.summary(i)) 
    
    #ESTUDIO DE LOS RESIDUOS
    #cálculo de los residuos
    y_train = y_train.flatten() 
    prediccion_train = modelo.predict(exog = X_train)
    residuos_train   = prediccion_train - y_train 
    
    #gráficas de los residuos
    
    #Se crean los ejes
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6)) 
    
    #grafico valor de prediccion vs valor real
    axes[0, 0].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.4, color='blue')
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
                    'k--', color = 'grey', lw=3)
    axes[0, 0].set_title('Prediccion vs Valor Real', fontsize = 10)
    axes[0, 0].set_xlabel('Real')
    axes[0, 0].set_ylabel('Predicción')
    axes[0, 0].tick_params(labelsize = 6)

    #grafico de residuos ajustados
    axes[0, 1].scatter(list(range(len(y_train))), residuos_train,
                       edgecolors=(0, 0, 0), alpha = 0.4, color='blue')
    axes[0, 1].axhline(y = 0, linestyle = '--', color = 'grey', lw=2)
    axes[0, 1].set_title('Residuos del modelo', fontsize = 10)
    axes[0, 1].set_xlabel('ID')
    axes[0, 1].set_ylabel('Residuo')
    axes[0, 1].tick_params(labelsize = 6)
    
    #grafico de distribución y densidad de los residuos
    sns.histplot(
        data    = residuos_train,
        stat    = "density",
        kde     = True,
        line_kws= {'linewidth': 1},
        color   = "blue",
        alpha   = 0.3,
        ax      = axes[0, 2]
    )

    axes[0, 2].set_title('Distribución residuos', fontsize = 10)
    axes[0, 2].set_xlabel("Residuos")
    axes[0, 2].tick_params(labelsize = 6)

    # Se eliminan los ejes vacíos
    fig.delaxes(axes[1,0])
    fig.delaxes(axes[1,1])
    fig.delaxes(axes[1,2])

    fig.tight_layout()
    plt.subplots_adjust(top=0.9);

    #TEST SHAPIRO SOBRE LOOS RESIDUOS DEL ENTRENAMIENTO
    shapiro_test = stats.shapiro(residuos_train)
    print([i], shapiro_test)
    
    #PREDICCIONES DEL MODELO
    predicciones = modelo.get_prediction(exog = X_train).summary_frame(alpha=0.05)#PREDICE EL MOODELO

    X_test = sm.add_constant(X_test, prepend=True)#PREDICE LOS VALORES CON EL CONJUNTO DE TEST
    predicciones = modelo.predict(exog = X_test)
    
    #CALCULO DEL RMSE PARA EL CONJUNTO DE TEST
    rmse = mean_squared_error(
            y_true  = y_test,
            y_pred  = predicciones,
            squared = False
           )
    print("")
    print([i],f"RMSE: {rmse}")


# In[29]:


#ESTIMACIÓN DEL MODELO Y RESIDUOS PERÍODO 1

items1= subset1[['Utilities', 'HC','EN']]

for i in items1:
    X = subset1[['SP', 'primaESG']]
    y = items1[i]


    X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )
    X_train = sm.add_constant(X_train, prepend=True)
    modelo = sm.OLS(endog=y_train, exog=X_train,)
    modelo = modelo.fit()
    print("Modelo", [i], modelo.summary(i))
    
   #ESTUDIO DE LOS RESIDUOS
    #cálculo de los residuos
    y_train = y_train.flatten() 
    prediccion_train = modelo.predict(exog = X_train)
    residuos_train   = prediccion_train - y_train
    
    #gráficas de los residuos
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))

    axes[0, 0].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.4, color='blue')
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
                    'k--', color = 'grey', lw=3)
    axes[0, 0].set_title('Prediccion vs Valor Real', fontsize = 10)
    axes[0, 0].set_xlabel('Real')
    axes[0, 0].set_ylabel('Predicción')
    axes[0, 0].tick_params(labelsize = 6)

    axes[0, 1].scatter(list(range(len(y_train))), residuos_train,
                       edgecolors=(0, 0, 0), alpha = 0.4, color='blue')
    axes[0, 1].axhline(y = 0, linestyle = '--', color = 'grey', lw=2)
    axes[0, 1].set_title('Residuos del modelo', fontsize = 10)
    axes[0, 1].set_xlabel('ID')
    axes[0, 1].set_ylabel('Residuo')
    axes[0, 1].tick_params(labelsize = 6)

    sns.histplot(
        data    = residuos_train,
        stat    = "density",
        kde     = True,
        line_kws= {'linewidth': 1},
        color   = "blue",
        alpha   = 0.3,
        ax      = axes[0, 2]
    )

    axes[0, 2].set_title('Distribución residuos', fontsize = 10)
    axes[0, 2].set_xlabel("Residuos")
    axes[0, 2].tick_params(labelsize = 6)

    # Se eliminan los ejes vacíos
    fig.delaxes(axes[1,0])
    fig.delaxes(axes[1,1])
    fig.delaxes(axes[1,2])

    fig.tight_layout()
    plt.subplots_adjust(top=0.9);

    #SHAPIRO
    shapiro_test = stats.shapiro(residuos_train)
    print([i], shapiro_test)
    
    #PREDICCIONES

    predicciones = modelo.get_prediction(exog = X_train).summary_frame(alpha=0.05)

    X_test = sm.add_constant(X_test, prepend=True)
    predicciones = modelo.predict(exog = X_test)
    rmse = mean_squared_error(
            y_true  = y_test,
            y_pred  = predicciones,
            squared = False
           )
    print("")
    print([i],f"RMSE: {rmse}")


# In[30]:


#ESTIMACIÓN DEL MODELO Y RESIDUOS PERÍODO 2

items2= subset2[['Utilities', 'HC','EN']]

for i in items2:
    X = subset2[['SP', 'primaESG']]
    y = items2[i]

    X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )
    X_train = sm.add_constant(X_train, prepend=True)
    modelo = sm.OLS(endog=y_train, exog=X_train,)
    modelo = modelo.fit()
    print("Modelo", [i], modelo.summary(i))
    
    #ESTUDIO DE LOS RESIDUOS
    #cálculo de los residuos
    y_train = y_train.flatten() 
    prediccion_train = modelo.predict(exog = X_train)
    residuos_train   = prediccion_train - y_train
    
    #gráficas de los residuos
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))

    axes[0, 0].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.4, color='blue')
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
                    'k--', color = 'grey', lw=3)
    axes[0, 0].set_title('Prediccion vs Valor Real', fontsize = 10)
    axes[0, 0].set_xlabel('Real')
    axes[0, 0].set_ylabel('Predicción')
    axes[0, 0].tick_params(labelsize = 6)

    axes[0, 1].scatter(list(range(len(y_train))), residuos_train,
                       edgecolors=(0, 0, 0), alpha = 0.4, color='blue')
    axes[0, 1].axhline(y = 0, linestyle = '--', color = 'grey', lw=2)
    axes[0, 1].set_title('Residuos del modelo', fontsize = 10)
    axes[0, 1].set_xlabel('ID')
    axes[0, 1].set_ylabel('Residuo')
    axes[0, 1].tick_params(labelsize = 6)

    sns.histplot(
        data    = residuos_train,
        stat    = "density",
        kde     = True,
        line_kws= {'linewidth': 1},
        color   = "blue",
        alpha   = 0.3,
        ax      = axes[0, 2]
    )

    axes[0, 2].set_title('Distribución residuos', fontsize = 10)
    axes[0, 2].set_xlabel("Residuos")
    axes[0, 2].tick_params(labelsize = 6)

    # Se eliminan los ejes vacíos
    fig.delaxes(axes[1,0])
    fig.delaxes(axes[1,1])
    fig.delaxes(axes[1,2])

    fig.tight_layout()
    plt.subplots_adjust(top=0.9);

    #SHAPIRO
    shapiro_test = stats.shapiro(residuos_train)
    print([i], shapiro_test)
    
    #PREDICCIONES
    predicciones = modelo.get_prediction(exog = X_train).summary_frame(alpha=0.05)

    X_test = sm.add_constant(X_test, prepend=True)
    predicciones = modelo.predict(exog = X_test)
    rmse = mean_squared_error(
            y_true  = y_test,
            y_pred  = predicciones,
            squared = False
           )
    print("")
    print([i],f"RMSE: {rmse}")



# In[31]:


#ESTIMACIÓN MODELO Y RESIDUOS PERÍODO 3

items3= subset3[['Utilities', 'HC','EN']]

for i in items3:
    X = subset3[['SP', 'primaESG']]
    y = items3[i]

    X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )
    X_train = sm.add_constant(X_train, prepend=True)
    modelo = sm.OLS(endog=y_train, exog=X_train,)
    modelo = modelo.fit()
    print("Modelo", [i], modelo.summary(i))
    
   #ESTUDIO DE LOS RESIDUOS
    #cálculo de los residuos
    y_train = y_train.flatten() 
    prediccion_train = modelo.predict(exog = X_train)
    residuos_train   = prediccion_train - y_train
    
    #gráficas de los residuos
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))

    axes[0, 0].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.4, color='blue')
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
                    'k--', color = 'grey', lw=3)
    axes[0, 0].set_title('Prediccion vs Valor Real', fontsize = 10)
    axes[0, 0].set_xlabel('Real')
    axes[0, 0].set_ylabel('Predicción')
    axes[0, 0].tick_params(labelsize = 6)

    axes[0, 1].scatter(list(range(len(y_train))), residuos_train,
                       edgecolors=(0, 0, 0), alpha = 0.4, color='blue')
    axes[0, 1].axhline(y = 0, linestyle = '--', color = 'grey', lw=2)
    axes[0, 1].set_title('Residuos del modelo', fontsize = 10)
    axes[0, 1].set_xlabel('ID')
    axes[0, 1].set_ylabel('Residuo')
    axes[0, 1].tick_params(labelsize = 6)

    sns.histplot(
        data    = residuos_train,
        stat    = "density",
        kde     = True,
        line_kws= {'linewidth': 1},
        color   = "blue",
        alpha   = 0.3,
        ax      = axes[0, 2]
    )

    axes[0, 2].set_title('Distribución residuos', fontsize = 10)
    axes[0, 2].set_xlabel("Residuos")
    axes[0, 2].tick_params(labelsize = 6)

    # Se eliminan los ejes vacíos
    fig.delaxes(axes[1,0])
    fig.delaxes(axes[1,1])
    fig.delaxes(axes[1,2])

    fig.tight_layout()
    plt.subplots_adjust(top=0.9);

    #SHAPIRO
    shapiro_test = stats.shapiro(residuos_train)
    print([i], shapiro_test)

    #PREDICCIONES
    predicciones = modelo.get_prediction(exog = X_train).summary_frame(alpha=0.05)


    X_test = sm.add_constant(X_test, prepend=True)
    predicciones = modelo.predict(exog = X_test)
    rmse = mean_squared_error(
            y_true  = y_test,
            y_pred  = predicciones,
            squared = False
           )
    print("")
    print([i],f"El error (rmse) de test es: {rmse}")


# In[32]:


#ESTIIMACIÓN MODELO Y RESIDUOS PERÍODO 4

items4= subset4[['Utilities', 'HC','EN']]
for i in items4:
    X = subset4[['SP', 'primaESG']]
    y = items4[i]

    X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )
    X_train = sm.add_constant(X_train, prepend=True)
    modelo = sm.OLS(endog=y_train, exog=X_train,)
    modelo = modelo.fit()
    print("Modelo", [i], modelo.summary(i))
    
    #ESTUDIO DE LOS RESIDUOS
    #cálculo de los residuos
    y_train = y_train.flatten() 
    prediccion_train = modelo.predict(exog = X_train)
    residuos_train   = prediccion_train - y_train
    
    #gráficas de los residuos
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))

    axes[0, 0].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.4, color='blue')
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
                    'k--', color = 'grey', lw=3)
    axes[0, 0].set_title('Prediccion vs Valor Real', fontsize = 10)
    axes[0, 0].set_xlabel('Real')
    axes[0, 0].set_ylabel('Predicción')
    axes[0, 0].tick_params(labelsize = 6)

    axes[0, 1].scatter(list(range(len(y_train))), residuos_train,
                       edgecolors=(0, 0, 0), alpha = 0.4, color='blue')
    axes[0, 1].axhline(y = 0, linestyle = '--', color = 'grey', lw=2)
    axes[0, 1].set_title('Residuos del modelo', fontsize = 10)
    axes[0, 1].set_xlabel('ID')
    axes[0, 1].set_ylabel('Residuo')
    axes[0, 1].tick_params(labelsize = 6)

    sns.histplot(
        data    = residuos_train,
        stat    = "density",
        kde     = True,
        line_kws= {'linewidth': 1},
        color   = "blue",
        alpha   = 0.3,
        ax      = axes[0, 2]
    )

    axes[0, 2].set_title('Distribución residuos', fontsize = 10)
    axes[0, 2].set_xlabel("Residuos")
    axes[0, 2].tick_params(labelsize = 6)

    # Se eliminan los ejes vacíos
    fig.delaxes(axes[1,0])
    fig.delaxes(axes[1,1])
    fig.delaxes(axes[1,2])

    fig.tight_layout()
    plt.subplots_adjust(top=0.9);

    #SHAPIRO
    shapiro_test = stats.shapiro(residuos_train)
    print([i], shapiro_test)

    #PREDICCIONES
    predicciones = modelo.get_prediction(exog = X_train).summary_frame(alpha=0.05)


    X_test = sm.add_constant(X_test, prepend=True)
    predicciones = modelo.predict(exog = X_test)
    rmse = mean_squared_error(
            y_true  = y_test,
            y_pred  = predicciones,
            squared = False
           )
    print("")
    print([i],f"RMSE: {rmse}")


# In[33]:


#ESTIMACIÓN MODELO Y RESIDUOS PERÍODO 5

items5= subset5[['Utilities', 'HC','EN']]
for i in items5:
    X = subset5[['SP', 'primaESG']]
    y = items5[i]


    X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )
    X_train = sm.add_constant(X_train, prepend=True)
    modelo = sm.OLS(endog=y_train, exog=X_train,)
    modelo = modelo.fit()
    print("Modelo", [i], modelo.summary(i))
    
    #ESTUDIO DE LOS RESIDUOS
    #cálculo de los residuos
    y_train = y_train.flatten() 
    prediccion_train = modelo.predict(exog = X_train)
    residuos_train   = prediccion_train - y_train
    
    #gráficas de los residuos
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))

    axes[0, 0].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.4, color='blue')
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
                    'k--', color = 'grey', lw=3)
    axes[0, 0].set_title('Prediccion vs Valor Real', fontsize = 10)
    axes[0, 0].set_xlabel('Real')
    axes[0, 0].set_ylabel('Predicción')
    axes[0, 0].tick_params(labelsize = 6)

    axes[0, 1].scatter(list(range(len(y_train))), residuos_train,
                       edgecolors=(0, 0, 0), alpha = 0.4, color='blue')
    axes[0, 1].axhline(y = 0, linestyle = '--', color = 'grey', lw=2)
    axes[0, 1].set_title('Residuos del modelo', fontsize = 10)
    axes[0, 1].set_xlabel('ID')
    axes[0, 1].set_ylabel('Residuo')
    axes[0, 1].tick_params(labelsize = 6)

    sns.histplot(
        data    = residuos_train,
        stat    = "density",
        kde     = True,
        line_kws= {'linewidth': 1},
        color   = "blue",
        alpha   = 0.3,
        ax      = axes[0, 2]
    )

    axes[0, 2].set_title('Distribución residuos', fontsize = 10)
    axes[0, 2].set_xlabel("Residuos")
    axes[0, 2].tick_params(labelsize = 6)

    # Se eliminan los ejes vacíos
    fig.delaxes(axes[1,0])
    fig.delaxes(axes[1,1])
    fig.delaxes(axes[1,2])

    fig.tight_layout()
    plt.subplots_adjust(top=0.9);

    #SHAPIRO
    shapiro_test = stats.shapiro(residuos_train)
    print([i], shapiro_test)

    predicciones = modelo.get_prediction(exog = X_train).summary_frame(alpha=0.05)

    X_test = sm.add_constant(X_test, prepend=True)
    predicciones = modelo.predict(exog = X_test)
    rmse = mean_squared_error(
            y_true  = y_test,
            y_pred  = predicciones,
            squared = False
           )
    print("")
    print([i],f"RMSE: {rmse}")


# In[34]:


#ESTIMACIÓN MODELO Y RESIDUOS PERÍODO 6 

items6= subset6[['Utilities', 'HC','EN']]


for i in items6:
    X = subset6[['SP', 'primaESG']]
    y = items6[i]

    X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )
    X_train = sm.add_constant(X_train, prepend=True)
    modelo = sm.OLS(endog=y_train, exog=X_train,)
    modelo = modelo.fit()
    print("Modelo", [i], modelo.summary(i))
    
    #ESTUDIO DE LOS RESIDUOS
    #cálculo de los residuos
    y_train = y_train.flatten() 
    prediccion_train = modelo.predict(exog = X_train)
    residuos_train   = prediccion_train - y_train
    
    #gráficas de los residuos
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))

    axes[0, 0].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.4, color='blue')
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
                    'k--', color = 'grey', lw=3)
    axes[0, 0].set_title('Prediccion vs Valor Real', fontsize = 10)
    axes[0, 0].set_xlabel('Real')
    axes[0, 0].set_ylabel('Predicción')
    axes[0, 0].tick_params(labelsize = 6)

    axes[0, 1].scatter(list(range(len(y_train))), residuos_train,
                       edgecolors=(0, 0, 0), alpha = 0.4, color='blue')
    axes[0, 1].axhline(y = 0, linestyle = '--', color = 'grey', lw=2)
    axes[0, 1].set_title('Residuos del modelo', fontsize = 10)
    axes[0, 1].set_xlabel('ID')
    axes[0, 1].set_ylabel('Residuo')
    axes[0, 1].tick_params(labelsize = 6)

    sns.histplot(
        data    = residuos_train,
        stat    = "density",
        kde     = True,
        line_kws= {'linewidth': 1},
        color   = "blue",
        alpha   = 0.3,
        ax      = axes[0, 2]
    )

    axes[0, 2].set_title('Distribución residuos', fontsize = 10)
    axes[0, 2].set_xlabel("Residuos")
    axes[0, 2].tick_params(labelsize = 6)

    # Se eliminan los ejes vacíos
    fig.delaxes(axes[1,0])
    fig.delaxes(axes[1,1])
    fig.delaxes(axes[1,2])

    fig.tight_layout()
    plt.subplots_adjust(top=0.9);

    #SHAPIRO
    shapiro_test = stats.shapiro(residuos_train)
    print([i], shapiro_test)

    #PREDICCIONES
    predicciones = modelo.get_prediction(exog = X_train).summary_frame(alpha=0.05)

    X_test = sm.add_constant(X_test, prepend=True)
    predicciones = modelo.predict(exog = X_test)
    rmse = mean_squared_error(
            y_true  = y_test,
            y_pred  = predicciones,
            squared = False
           )
    print("")
    print([i],f"RMSE: {rmse}")


# In[ ]:




