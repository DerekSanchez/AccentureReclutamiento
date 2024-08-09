
import pandas as pd
from sklearn.feature_selection import chi2, SelectKBest 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

#Funcion para analizar unicidad por variable
#Input es un dataframe
#Output es un diccionario con el nombre de columnas y el recuento de obs unicas
def UnicosFeature(df):
    RecuentoUnico = {}
    for column in df.columns:
        RecuentoUnico[column] = df[column].nunique()
    return RecuentoUnico

#******************Tratamiento Variables Numericas******************

#Funcion para hacer cambio de observaciones no numericas a NaN
#Input es un df y columnas que por contexto sabemos que son numericas
#Output es el df con obs no numericas transformadas a NaN
def CambioNoNumerico(df, columnas):
    df = df.copy()
    
    for columna in columnas:
        df[columna] = pd.to_numeric(df[columna], errors = 'coerce')
    return df


#Funcion para rellenar los vacios numéricos
#Input es un df y columnas que por contexto sabemos que son numericas
#Output es el df con NaN sustituido por media del feature
def RellenoNA(df, columnas): 
    df = df.copy()
    
    for columna in columnas:
        df[columna] = df[columna].fillna(df[columna].mean())
    return df


#Funcion para cambiar tipo de dato de variables numericas float
#Input es un df y columnas que por contexto sabemos que son numericas
#Output es un df con dichas columnas en formato float

def ToFloat(df, columnas):
    df = df.copy()
    
    for columna in columnas:
        df[columna] = df[columna].astype(float)
    return df
    
    
#******************Tratamiento Variables Categoricas******************

#Funcion para simplificar clases categoricas
#Input es un df y columnas categoricas con varias categorias que se pueden simplificar transformando a binarias
#Output el df con las columnas categoricas convertidas a binarias
def aBinario(df, columns):
    def convert(value):
        if value == 'Yes':
            return 'Yes'
        else:
            return 'No'
    for column in columns:
        df[column] = df[column].apply(convert)
    
    return df


#Funcion para transformar variables binarias a 0 y 1
#Input es un df y columnas categóricas
#Output es esas mismas columnas pero con 0s y 1s
def EncodingBinario(df, columnas):
    df = df.copy()
    
    ColsCatBinaria = [col for col in columnas if df[col].nunique() == 2]
    
    for columna in ColsCatBinaria:
        valores_unicos = df[columna].unique()
        mapeo = {valores_unicos[0]:0, valores_unicos[1]:1}
        df[columna] = df[columna].map(mapeo)
    return df


#Función para hacer one hot encoding en features categóricos no-ordinales y no-binarias
#Input es un df y columnas que por contexto sabemos que son categóricas
#Outpt es un dataframe con columnas indicadas transformadas a one hot enconding
def OneHotEncode(df, columnas):
    df = df.copy()
    
    ColsCatNoBinaria = [col for col in columnas if df[col].nunique() != 2]
    for columna in ColsCatNoBinaria:
        dummies = pd.get_dummies(df[columna], prefix = columna)
        df = pd.concat([df, dummies], axis = 1)
        df = df.drop(columna, axis = 1)
    return df

#Funcion que realiza prueba de independencia entre el target y variables categoricas en one hot encoding
#Input es un df y columnas que por contexto sabemos que son categóricas
#Output es una lista de features que exceden el threshold de significancia y seran excluidos
def Chi2Filtering(df, columnasCat, Y, pvalue_threshold):
    X = OneHotEncode(df[columnasCat], columnasCat)
    
    PruebaChiCuad = SelectKBest(chi2, k = 'all')
    PruebaChiCuad.fit(X, Y)
    pvalues = PruebaChiCuad.pvalues_

    pvalues_df = pd.DataFrame({'Feature': X.columns, 'p-value': pvalues})
    pvalues_df.sort_values('p-value', inplace = True, ascending= False)
    pvalues_df
    
    FeaturesExcluidosDummies = pvalues_df[pvalues_df['p-value'] > pvalue_threshold]['Feature'].tolist()
    FeaturesExcluidos= list(set([s.split('_')[0] for s in FeaturesExcluidosDummies]))
    
    #print("P values por variable")
    #print(pvalues_df)
    print("Features a excluir: ", FeaturesExcluidos)
    return FeaturesExcluidos



def escala_data(data, method="standard"):

    if method == "min-max":
        scaler = MinMaxScaler()
    elif method == "standard":
        scaler = StandardScaler()
    elif method == "robust":
        scaler = RobustScaler()
    elif method == "max-abs":
        scaler = MaxAbsScaler()
    else:
        raise ValueError(f"Método de escalado '{method}' no es válido. Elija entre 'min-max', 'standard', 'robust', o 'max-abs'.")
    
    # Ajustar y transformar los datos
    scaled_data = pd.DataFrame(scaler.fit_transform(data), columns = data.columns)
    
    return scaled_data

