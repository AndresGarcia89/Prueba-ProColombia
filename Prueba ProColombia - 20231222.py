# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 07:01:22 2023

@author: andrs
"""

import pandas as pd
import seaborn as sns
import numpy as np
import cpi
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from datetime import date
import os

os.chdir("D:\Documents")

enlace_tablas_extra='https://drive.google.com/uc?id={}'.format('1EM6F8qqIay-KEINFi6gQU8piuFCFT2hw')
pd.ExcelFile(enlace_tablas_extra).sheet_names
tabla_municipios=pd.read_excel(enlace_tablas_extra,sheet_name="Correlativa municipios")
tabla_paises=pd.read_excel(enlace_tablas_extra,sheet_name="Correlativa ISO Country")
tabla_motivos=pd.read_excel(enlace_tablas_extra,sheet_name="Correlativa motivo de viaje")

enlace_datos='https://drive.google.com/uc?id={}'.format('1-4Ft5xlIFnZ-XZzBxlHlpchX8TxlBg3G')
tabla_datos=pd.read_csv(enlace_datos,compression="zip",sep="|")

# Tras importar los datos, se revisa la tendencia de turistas totales en el país
tabla_datos.groupby("Año").sum()["Cantidad turistas"]

# Se observa una caída importante en 2020 y 2021, por motivos de pandemia
# Debido a lo excepcionales que son estos años, no se incluirán en los pronósticos
tabla_sin_pandemia=tabla_datos[tabla_datos["Año"].isin([2015,2016,2017,2018,2019,2022,2023])]

# Ahora, vemos la agrupación de turistas por mes y año
tabla_sin_pandemia.groupby(["Año","Mes"]).sum()["Cantidad turistas"]

# Los datos de 2023 van hasta agosto.
plt.figure(figsize=(12,8))
sns.lineplot(x="Mes",y="Cantidad turistas",estimator="sum",hue="Año",palette="bright",data=tabla_datos,errorbar=None).set_title("Turistas por mes")
sns.lineplot(x="Año",y="Cantidad turistas",estimator="sum",palette="bright",data=tabla_datos,errorbar=None).set_title("Turistas por año")

# Con el objetivo de predecir, lo mejor será tomar los datos hasta agosto, y ver la tendencia
tabla_sin_pandemia_agosto=tabla_sin_pandemia[tabla_sin_pandemia["Mes"].isin(np.arange(1,9,1))]
sns.lineplot(x="Año",y="Cantidad turistas",estimator="sum",palette="bright",data=tabla_sin_pandemia_agosto,errorbar=None).set_title("Turistas por año")

# Se utilizará una proyección sencilla, calculando el aumento interanual. 
# Para interpolar 2023, se usará la proporción de turistas por año entre septiembre y diciembre
tabla_sin_pandemia_agosto[tabla_sin_pandemia_agosto["Año"]!=2023]["Cantidad turistas"].sum()
tabla_sin_pandemia[tabla_sin_pandemia["Año"]!=2023]["Cantidad turistas"].sum()
proporcion_agosto=tabla_sin_pandemia_agosto[tabla_sin_pandemia_agosto["Año"]!=2023]["Cantidad turistas"].sum()/tabla_sin_pandemia[tabla_sin_pandemia["Año"]!=2023]["Cantidad turistas"].sum()

proyeccion_2023=tabla_sin_pandemia[tabla_sin_pandemia["Año"]==2023]["Cantidad turistas"].sum()/proporcion_agosto

turistas_anuales=tabla_datos.groupby("Año").sum()["Cantidad turistas"]
# Con estos datos, se buscará proyectar el aumento de turistas.

turistas_anuales.drop(labels=[2020,2021],inplace=True)
turistas_anuales[2023]=round(proyeccion_2023,0)
turistas_anuales
crec_turistas=pow(turistas_anuales[2023]/turistas_anuales[2015],1/6)-1
turistas_anuales[2024]=round(turistas_anuales[2023]*(1+crec_turistas),0)

# Con esto calculado, ahora se buscarán identificar los orígenes de mayor crecimiento en 2023
turistas_por_pais=pd.DataFrame(tabla_sin_pandemia_agosto.groupby(["Código ISO país","Año"]).sum()["Cantidad turistas"])
turistas_por_pais=turistas_por_pais.unstack()
turistas_por_pais.sort_values("2023",ascending=False)

ax=sns.barplot(turistas_anuales).set_title("Total de turistas anuales con proyección 2024")
for i in ax.containers:
    ax.bar_label(i,)
