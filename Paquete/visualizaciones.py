import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#Funcion para generar columnas apiladas y evaluar desbalanceo de clases
#Input es un df y columnas categoricas
#Output son graficas de columnas indicadas
def PlotDesbalanceoClass(df, columnas):
    df = df.copy()
    
    for columna in columnas:
        
        prop_df = df[columna].value_counts(normalize = True).reset_index()
        prop_df.columns = [columna, 'proporcion']
        
        plt.figure(figsize = (5,3))
        plt.bar(prop_df[columna], prop_df['proporcion'], color = ['skyblue','salmon','lightgreen','lightcoral'])        
        plt.title(f'Distribucion de cada categoria para {columna}')
        plt.xlabel('Categorias')
        plt.ylabel('proporcion')
        plt.ylim(0,1)
        plt.xticks(rotation = 0)
        plt.show()

#Funcion para generar una serie de boxplots para features numericos
#Input es un df con columnas numericas, variable objetivo y el nombre de la variable objetivo
#Output es un boxplot por cada feature
def BoxPlotIterativo(X_numericas, y_target, y_nombre):
    
    sns.set_palette("Purples")
    
    Data_Temp = pd.concat([X_numericas, y_target], axis = 1)

    for ColumnaNumerica in X_numericas.columns:
        plt.figure(figsize = (4,3))
        sns.boxplot(x = y_target[y_nombre], y = ColumnaNumerica, data = Data_Temp)
        plt.show()

#Funcion que grafica histogramas
#Input es un dataframe con features numericos
#Output es una serie de histogramas
def HistogramasIndividuales(df):
    df.hist(bins = 15, figsize = (5,4), color = 'purple', grid = False)


#Funcion que grafica multiples distribuciones suavizadas con kernell
#Input es un dataframe y una lista de columnas a graficas
#Output es un solo grafico con todas las distribuciones indicadas
def DensidadSuavizadaSimultanea(df, columns, palette="Purples", bw_adjust=0.5, linewidth=2, sizeL = 10, sizeW = 6):
    
    # Filtrar las columnas que se desean graficar
    df_filtered = df[columns]
    
    # Crear figura
    plt.figure(figsize=(sizeL, sizeW))
    
    # Definir paleta de colores
    color_palette = sns.color_palette(palette, len(columns))
    
    # Graficar cada columna
    for i, column in enumerate(columns):
        sns.kdeplot(df_filtered[column].dropna(), bw_adjust=bw_adjust, label=column, color=color_palette[i], linewidth=linewidth)
    
    # Mostrar leyenda
    plt.legend()
    
    # Eliminar grid, títulos y etiquetas de los ejes
    plt.grid(False)
    plt.title('')
    plt.xlabel('')
    plt.ylabel('')
    
    # Mostrar gráfica
    plt.show()
    
    
#Heatmap de correlaciones
#Input es un df de features y el target
#Output es una matriz de correlaciones con heatmap
def HeatmapCorr(X, y, annotacion=True, mostrar_ejes=True, mostrar_barra_color=True, largo = 5, ancho = 4):
    # Combinar los datos del target y las features
    Data_Temp = pd.concat([y, X], axis=1)
    # Calcular la matriz de correlaciones
    corr = Data_Temp.corr()

    # Crear la figura para el heatmap
    plt.figure(figsize=(largo, ancho))
    
    # Generar el heatmap con los parámetros adicionales
    sns.heatmap(
        corr, 
        annot=annotacion, 
        vmin=-1.0, 
        cmap='Purples', 
        cbar=mostrar_barra_color,
        xticklabels=mostrar_ejes,
        yticklabels=mostrar_ejes
    )
    
    # Mostrar el heatmap
    plt.show()