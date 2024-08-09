#Librerias
import os
import pandas as pd

#Funcion para generar ruta de lectura. Asume estructura estandar de desarrollo
#Input es el nombre del archivo y su extension
#Output es la ruta para leerlo
def GetRutaInput(archivo):
    #Ruta al codigo
    ruta_actual = os.getcwd()
    
    #Ruta a carpeta global de proyecto
    ruta_afuera = os.path.dirname(ruta_actual)
    
    #Genera ruta de extracción de inputs de carpeta Inputs
    InputDataPath = os.path.join(ruta_afuera, 'Inputs', archivo)
    
    return InputDataPath

#Funcion para leer Exceles
#Input la ruta del archivo
#Output es un dataframe consolidado
def LecturaData(InputDataPath):
    
    #Leer tablas de insumo
    Data1 = pd.read_excel(InputDataPath, sheet_name = 'Charges')
    Data2 = pd.read_excel(InputDataPath, sheet_name = 'Other data')
    Data3 = pd.read_excel(InputDataPath, sheet_name = 'Churn')
    
    #Consolidación de tablas
    Data = pd.merge(Data1, Data2, on = 'customerID', how = 'inner')
    Data = pd.merge(Data, Data3, on = 'customerID', how = 'inner')
    pd.set_option('display.max_columns', None)
    
    return Data





    
    