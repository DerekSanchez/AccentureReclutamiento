#Modelo Churn

Modelo supervisado de Machine Learning de clasificacion binaria
Utilizado para clasificar a clientes de micro creditos de una fintech
Pretende estimar la probabilidad de default

## Tabla de Contenidos
- [Instalación]
- [Uso]
- [Estructura del Proyecto]


##Instalación

se necesita tener instaladas las siguientes librerias:
1. pandas
2. numpy
3. matplotlib
4. seaborn
5. sckit-learn
6. os

##Uso

El archivo tiene varios scripts con modulos para cada paso de un proceso tradicional de modelado de ML
(lectura de datos, visualizaciones, preprocessing, modelaje)
estos scripts contienen unicamente funciones, que se llaman en otro archivo (notebook)
en este archivo se ejectutan las funciones previamente mencionadas, con inputs

##Estructura 

Proyecto/
│
├── Inputs/                     #Inputs
│   └── Datos.xlsx
│
├── mi_paquete/               #folder con modulos
│   ├── __init__.py
│   ├── carga_datos.py
│   ├── preprocesamiento.py
│   ├── visualizaciones.py
│   └── modelaje.py
│
├── notebooks/                #Notebooks de Jupyter
│   └── Churn Model.ipynb     #Modelo principal, donde se aplican las funciones de los modulos a la data de input
├── setup.py                  #Script de configuración del paquete
├── .bumpversion.cfg          #Configuración de bumpversion
└── README.md                 #Este archivo
