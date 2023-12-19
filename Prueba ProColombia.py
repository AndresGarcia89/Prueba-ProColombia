# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 09:23:00 2023

@author: andrs
"""

# Primero, se importan los archivos dedsde la carpeta de Google Drive en la cual están alojados
# El primer archivo traído: el diccionario de datos
import pandas as pd
import seaborn as sns
import numpy as np
import cpi
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from datetime import date
import os

os.chdir("D:\Documents")

enlace_diccionario='https://drive.google.com/uc?id={}'.format('1vCqe1ALuSaFFk6Layc0zc490fdbXYqPK')
pd.ExcelFile(enlace_diccionario).sheet_names
diccionario=pd.read_excel(enlace_diccionario)

# Tras revisar el diccionario de datos, y uno de los archivos planos (RUES.txt), se decidió
# Que lo mejor sería asumir que todas las variables son caracteres, excepto siete variables
# Estas siete variables deben especificarse como números decimales. Para ello, se creó
# Un diccionario de tipos de variables, usando los nombres de variables en "diccionario"
# ESte diccionario de tipos se llama "tipos_variables"

tipos_variables = {'Activos': float, 'Ingresos operacionales': float,'Antigüedad empresa': float,'Expo 2022': float,'Expo prom ult 5 años': float,'Var. Expo 2022': float, 'TCAC expo ult 5 años': float}
tipos_variables.update({col: str for col in diccionario["Nombre columna"] if col not in tipos_variables})

# Tras definir el diccionario de tipos de variables, se importan los demás conjuntos de datos
# Se abrieron los archivos planos para observar el separador y el caracter de decimales
# Hecho esto, se ajustaron los parámetros de importación

enlace_expo_nme='https://drive.google.com/uc?id={}'.format('1hrWN6IlK0CjaRaAlum1Z3ZzbNVcXZW3L')
pd.ExcelFile(enlace_expo_nme).sheet_names
expo_nme=pd.read_excel(enlace_expo_nme)
enlace_directorio='https://drive.google.com/uc?id={}'.format('1mMx09kYrT0fMfAE0PhUumhP3MbEE8pLW')
directorio=pd.read_table(enlace_directorio,sep='|',decimal=',',dtype=tipos_variables)
enlace_exportaciones='https://drive.google.com/uc?id={}'.format('1pDXqqYMHMJxcDwq_w_bBEvsaTquVIrKl')
exportaciones=pd.read_table(enlace_exportaciones,sep='|',decimal=',',dtype=tipos_variables)
enlace_supersociedades='https://drive.google.com/uc?id={}'.format('14g3sgApZccwHcv8pL4a6VqeKqIFA8GRz')
supersociedades=pd.read_table(enlace_supersociedades,sep='|',decimal=',',dtype=tipos_variables)
enlace_rues='https://drive.google.com/uc?id={}'.format('1ZIfvXo-4sXmkqOWzC3kqUp7zuFryGfBC')
rues=pd.read_table(enlace_rues,sep='|',decimal=',',dtype=tipos_variables)

# Viendo el conjunto de datos, se encontraron las siguientes observaciones:
# se halló que Antioquia y Atlántico no tienen el cero al inicio de su código departamental
# También, los municipios de esos departamentos carecen del cero al inicio
# Más importante aún, faltan los ceros iniciales de algunos códigos CIIU, como el "143".
# Para ello, se completarán los ceros faltantes en tres columnas
# Específicamente: en "Cod. Depto", si tiene 1 dígito, se agregará un cero al inicio
# En "Cod. Municipio", si tiene 4 dígitos, se agregará un cero al inicio
# En "CIIU Rev 4 principal", si tiene tres dígitos, se agregará un cero al inicio

directorio["Cod. Depto"]=directorio["Cod. Depto"].str.zfill(2)
exportaciones["Cod. Depto"]=exportaciones["Cod. Depto"].str.zfill(2)
rues["Cod. Depto"]=rues["Cod. Depto"].str.zfill(2)
supersociedades["Cod. Depto"]=supersociedades["Cod. Depto"].str.zfill(2)

directorio["Cod. Municipio"]=directorio["Cod. Municipio"].str.zfill(5)
exportaciones["Cod. Municipio"]=exportaciones["Cod. Municipio"].str.zfill(5)
rues["Cod. Municipio"]=rues["Cod. Municipio"].str.zfill(5)
supersociedades["Cod. Municipio"]=supersociedades["Cod. Municipio"].str.zfill(5)

directorio["CIIU Rev 4 principal"]=directorio["CIIU Rev 4 principal"].str.zfill(4)
exportaciones["CIIU Rev 4 principal"]=exportaciones["CIIU Rev 4 principal"].str.zfill(4)
rues["CIIU Rev 4 principal"]=rues["CIIU Rev 4 principal"].str.zfill(4)
supersociedades["CIIU Rev 4 principal"]=supersociedades["CIIU Rev 4 principal"].str.zfill(4)

# Hecho lo anterior, ahora es necesario comenzar el análisis.

# Primero, se crearán los modelos para verificar los resultados de las empresas
# Dado que las bases "exportaciones", "rues", "supersociedades" y "directorio"
# Tienen las mismas variables, se pueden fusionar en una única base
# Teniendo cuidado de eliminar duplicados, eso sí
empresas=pd.concat([directorio,exportaciones,rues,supersociedades],axis=0).drop_duplicates()

# Ahora con esta lista única de empresas, quedaron 50 mil registros
# Hay varios valores faltantes. De momento, estos serán rellenados con ceros
# Se asume que las empresas con valores faltantes tuvieron valores de cero en sus variables numéricas
empresas.fillna(0,inplace=True)

# Igualmente, es necesario eliminar empresas que no tengan ingresos operacionales
# Si una empresa no tiene ingresos operacionales, no es posible garantizar su continuidad

empresas=empresas[empresas["Ingresos operacionales"]>0]
# Tras este filtro, quedaron 27051 empresas

# Ahora, es necezario ver cuáles son las características de las empresas que sí tuvieron exportaciones
# Como criterio, se tomó el promedio de exportaciones en los últimos cinco años
empresas["Exportadoras"]=np.where(empresas["Expo prom ult 5 años"]>0,1,0)
empresas["Exportadoras"].sum()

# De las 27051 empresas en la base de datos con ingresos en el último año, 1554 realizaron exportaciones en los últimos cinco años
# A continuación, es necesario identificar las principales características de estas empresas
# Para ello, se grafican las proporciones de empresas exportadoras por diferentes categorías 
# Esto se realiza con múltiples gráficos de barras.
# Un valor de 0 indica que ninguna empresa de la categoría analizada exportó en los últimos cinco años
# Un valor de 1 indica que todas las empresas de la categoría analizada exportaron en los últimos cinco años

depto=sns.barplot(errorbar=None,x=empresas["Cod. Depto"],y=empresas["Exportadoras"],order=empresas.groupby("Cod. Depto")["Exportadoras"].mean().sort_values(ascending=False).index)
depto.set_xticklabels(depto.get_xticklabels(), rotation=45)
depto.set_title("Proporción de empresas exportadoras por departamento")
plt.show()

tamano=sns.barplot(errorbar=None,x=empresas["Tamaño empresa RUES"],y=empresas["Exportadoras"],order=empresas.groupby("Tamaño empresa RUES")["Exportadoras"].mean().sort_values(ascending=False).index)
tamano.set_xticklabels(tamano.get_xticklabels(), rotation=45)
tamano.set_title("Proporción de empresas exportadoras por tamaño")
plt.show()

val_agregado=sns.barplot(errorbar=None,x=empresas["Valor agregado empresa"],y=empresas["Exportadoras"],order=empresas.groupby("Valor agregado empresa")["Exportadoras"].mean().sort_values(ascending=False).index)
val_agregado.set_xticklabels(val_agregado.get_xticklabels(), rotation=90)
val_agregado.set_title("Proporción de empresas exportadoras por valor agregado")
plt.show()

cadena_segmentacion=sns.barplot(errorbar=None,x=empresas["Cadena segmentación"],y=empresas["Exportadoras"],order=empresas.groupby("Cadena segmentación")["Exportadoras"].mean().sort_values(ascending=False).index)
cadena_segmentacion.set_xticklabels(cadena_segmentacion.get_xticklabels(), rotation=90)
cadena_segmentacion.set_title("Proporción de empresas exportadoras por cadena de segmentación")
plt.show()


# De acuerdo con los resultados, se observa cómo las participaciones de empresas exportadoras varían según múltiples criterios
# Por departamento, destacan las empresas en Cundinamarca (25), Antioquia (05), Valle (76) y Bogotá (11)
# Por cadena de CIIU, hay potencial en las empresas de logística e infraestructura
# Por tamaño, existe mayor propensión a exportar en las empresas grandes y medianas
# Por valor agregado, destacan empresas con bienes de tecnología alta, media-alta y baja
# Curiosamente, son menos propensas a exportar las empresas que no son sucursales de una sociedad extranjera
# Por cadena de segmentación, son más propensas en "Sistema Moda" y "Químicos y Ciencias de la Vida"

# Hecho lo anterior, se buscarán empresas que cumplan con un conjunto de criterios
# Específicamente, se mirarán las empresas con mayor propensión a exportar en cuatro categorías
# Departamento: Cundinamarca, Antioquia, Bogotá, Valle
# Tamapo de la empresa: mediana o grande
# Valor agregado de la empresa: bienes de alta y media-alta tecnología
contacto=empresas[empresas["Cod. Depto"].isin(["05","11","76","25"]) & empresas["Tamaño empresa RUES"].isin(["Grande","Mediana"]) & empresas["Valor agregado empresa"].isin(["Bienes tecnología alta","Bienes tecnología media-alta"]) & empresas["Exportadoras"].isin([0])]
contacto.to_excel("contacto.xlsx",index=False)
# A partir de lo anterior, se obtuvo un listado de 13 empresas para contactar
# Cabe destacar que las decisiones dependen de la política sectorial

# Posteriormente, se graficará el comportamiento de las exportaciones no minero-energéticas

sns.lineplot(data=expo_nme,y="Expo_NME",x="Mes").set_title("Exportaciones NME a precios corrientes")
plt.show()
# A primera vista, la serie parece tener una tendencia.
# También se observa que la serie es mensual.
# Promedio para los totales de exportaciones por mes
expo_nme["Num_Mes"]=expo_nme.Mes.dt.month
expo_nme["Num_Anno"]=expo_nme.Mes.dt.year

# Duda importante: si estos valores son en dólares, estos son constantes o corrientes?
# Viendo los datos, y ante la tendencia, creo que no es descabellado asumir
# Que los precios están en dólares corrientes
# Se convirtieron a dólares constantes, utilizando datos del Bureau of Labor Statistics
# Para el IPC general de los Estados Unidos en las fechas analizadas
# https://pypi.org/project/cpi/1.0.0/

cpi.update()
expo_nme["IPC_USA"]=expo_nme["Mes"].dt.date.apply(lambda x: cpi.get(x))
expo_nme["NME_Constante"]=expo_nme.Expo_NME*cpi.get(date(2023,9,1))/expo_nme.IPC_USA

# Ahora que tengo precios constantes, se explora de nuevo la serie
sns.lineplot(data=expo_nme,y="NME_Constante",x="Mes").set_title("Exportaciones NME a precios constantes")
plt.show()

# Se observa cómo los promedios mensuales y los totales anuales presentan tendencias
# En los siguientes gráficos de barras
sns.barplot(x="Num_Mes",y="NME_Constante",data=expo_nme,errorbar=("ci",95)).set_title("Exportaciones promedio por mes")
plt.show()
sns.barplot(x="Num_Anno",y="NME_Constante",data=expo_nme,estimator="sum").set_title("Exportaciones totales por año")
plt.show()
# A partir de lo anterior, puedo observar una tendencia

adfuller(expo_nme[expo_nme.Num_Anno!=2023].NME_Constante)
# Curioso. La Dickey-Fuller aumentada dice que la serie es estacionaria. O sea, que no hay aumento en los valores
# Vamos a ver la existencia de autocorrelaciones

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(expo_nme.NME_Constante, lags=20)
plot_pacf(expo_nme.NME_Constante, lags=20)

# Parece que, a pesar de todo, se necesitan unas diferencias.
plot_acf(expo_nme.NME_Constante.diff().dropna(), lags=20)
plot_pacf(expo_nme.NME_Constante.diff().dropna(), lags=20)

plot_acf(expo_nme.NME_Constante.diff(periods=12).dropna(), lags=20)
plot_pacf(expo_nme.NME_Constante.diff(periods=12).dropna(), lags=20)
# Tras ver estos gráficos, se observa que los datos necesitan una diferencia estacionaria
# Y también les caería bien una diferencia estacional

# Viendo las series de autocorrelación y autocorrelación parccial, se puede especificar un ARIMA
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
modelo_export_nme1=sm.tsa.statespace.SARIMAX(expo_nme[expo_nme.Num_Anno!=2023].NME_Constante,order=(2, 1, 1),seasonal_order=(1,1,2,12))
res1=modelo_export_nme1.fit()
pred1=res1.predict(start=204,end=212,dynamic=True)
mse1 = np.sqrt(mean_squared_error(expo_nme[expo_nme.Num_Anno==2023].NME_Constante, pred1))

modelo_export_nme2=sm.tsa.statespace.SARIMAX(expo_nme[expo_nme.Num_Anno!=2023].NME_Constante,order=(2, 1, 0),seasonal_order=(1,1,2,12))
res2=modelo_export_nme2.fit()
pred2=res2.predict(start=204,end=212,dynamic=True)
mse2 = np.sqrt(mean_squared_error(expo_nme[expo_nme.Num_Anno==2023].NME_Constante, pred2))

modelo_export_nme3=sm.tsa.statespace.SARIMAX(expo_nme[expo_nme.Num_Anno!=2023].NME_Constante,order=(2, 1, 0),seasonal_order=(0,1,2,12))
res3=modelo_export_nme3.fit()
pred3=res3.predict(start=204,end=212,dynamic=True)
mse3 = np.sqrt(mean_squared_error(expo_nme[expo_nme.Num_Anno==2023].NME_Constante, pred3))

modelo_export_nme4=sm.tsa.statespace.SARIMAX(expo_nme[expo_nme.Num_Anno!=2023].NME_Constante,order=(1, 1, 1),seasonal_order=(1,1,1,12))
res4=modelo_export_nme4.fit()
pred4=res4.predict(start=204,end=212,dynamic=True)
mse4 = np.sqrt(mean_squared_error(expo_nme[expo_nme.Num_Anno==2023].NME_Constante, pred4))

print("RMSE 1: ",mse1,", RMSE 2: ",mse2,", RMSE 3: ",mse3,", RMSE 4: ",mse4)

# Después de ver los diferentes modelos, y tomando en cuenta tanto el RMSE
# Como significancia de las variables, y los valores de los criterios de inforamción AIC y BIC
# Se eligió el modelo 2, con especificaciones ARIMA(2,1,0)xSARIMA(1,1,2,12)

pred_final=res2.predict(start=204,end=227,dynamic=True)
# Fusión de la serie original con los valores pronosticados con el modelo, usando índices de filas
expo_nme2=expo_nme.merge(pred_final,how="outer",left_index=True,right_index=True)
# Gráfico combinado
expo_nme2[['NME_Constante','predicted_mean']].plot(figsize=(6,4)).set_title("Exportaciones NME a precios constantes, más pronóstico a 2024")


# Elegido este modelo, se obtuvieron los siguientes resultados.
# A destacar: se espera poco crecimiento en las exportaciones NME a 2024, en dólares constantes
# Asumiendo que no se den cambios en política para aumentar el total de las exportaciones