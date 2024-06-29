import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def procesar_y_graficar_dataset(archivo_csv):
    # Leer el archivo CSV
    df = pd.read_csv(archivo_csv, parse_dates=['fecha'], index_col='fecha')

    # Exploración de Datos Inicial
    print(f"\nDataset: {archivo_csv}")
    print("Datos iniciales:")
    print(df.head())
    print("\nDescripción de los datos:")
    print(df.describe())

    # Identificación de Outliers y Reemplazo
    Q1 = df['temperatura_promedio'].quantile(0.25)
    Q3 = df['temperatura_promedio'].quantile(0.75)
    IQR = Q3 - Q1

    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    outliers = (df['temperatura_promedio'] < limite_inferior) | (df['temperatura_promedio'] > limite_superior)

    for idx in df.index[outliers]:
        fecha_outlier = idx
        fecha_anterior = fecha_outlier - pd.DateOffset(days=1)
        fecha_posterior = fecha_outlier + pd.DateOffset(days=1)

        if fecha_anterior in df.index and fecha_posterior in df.index:
            temperatura_anterior = df.loc[fecha_anterior, 'temperatura_promedio']
            temperatura_posterior = df.loc[fecha_posterior, 'temperatura_promedio']
            df.loc[idx, 'temperatura_promedio'] = np.mean([temperatura_anterior, temperatura_posterior])

    # Visualización del Dataset Procesado con Outliers Reemplazados
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['temperatura_promedio'], label='Temperatura Promedio', color='blue')
    plt.title(f'Temperatura Promedio - {archivo_csv}')
    plt.xlabel('Fecha')
    plt.ylabel('Temperatura Promedio')
    plt.legend()
    plt.grid(True)
    plt.show()

    return df


# Lista de archivos CSV por año
archivos_csv = [
    'temperaturas_promedio_2017.csv',
    'temperaturas_promedio_2018.csv',
    'temperaturas_promedio_2019.csv',
    'temperaturas_promedio_2020.csv',
    'temperaturas_promedio_2021.csv',
    'temperaturas_promedio_2022.csv',
    'temperaturas_promedio_2023.csv'
]

# Lista para almacenar DataFrames procesados
dfs_procesados = []

# Procesar y graficar cada archivo CSV
for archivo in archivos_csv:
    df_procesado = procesar_y_graficar_dataset(archivo)
    dfs_procesados.append(df_procesado)

# Combinar todos los DataFrames en uno solo
df_final = pd.concat(dfs_procesados)
data = pd.concat(dfs_procesados)

# Exploración de Datos del DataFrame Combinado
print("\nDescripción de los datos combinados:")
print(df_final.describe())

# Ingeniería de Características
df_final['media_movil_7'] = df_final['temperatura_promedio'].rolling(window=7).mean().fillna(0)
df_final['media_movil_30'] = df_final['temperatura_promedio'].rolling(window=30).mean().fillna(0)

'''
# Calcular el ICEN (utilizando temperaturas promedio para la región Niño 1+2)
# Asegúrate de ajustar esto según los datos disponibles y su frecuencia
df_final['ICEN'] = df_final['temperatura_promedio'].rolling(window=90).mean().fillna(0)


# Añadir Categoría y Condición basadas en ICEN
def categorizar_ICEN(valor):
    if valor <= -1.4:
        return 'La Niña Fuerte'
    elif -1.4 < valor <= -1.2:
        return 'La Niña Moderado'
    elif -1.2 < valor <= -1.0:
        return 'La Niña Débil'
    elif -1.0 < valor <= 0.4:
        return 'Neutro'
    elif 0.4 < valor <= 1.0:
        return 'El Niño Débil'
    elif 1.0 < valor <= 1.7:
        return 'El Niño Moderado'
    elif 1.7 < valor <= 3.0:
        return 'El Niño Fuerte'
    elif valor > 3.0:
        return 'El Niño Muy Fuerte'


df_final['Categoria'] = df_final['ICEN'].apply(categorizar_ICEN)
'''

# Visualización del DataFrame Final
plt.figure(figsize=(14, 7))
plt.plot(df_final.index, df_final['temperatura_promedio'], label='Temperatura Promedio', color='blue')
plt.plot(df_final.index, df_final['media_movil_7'], label='media_movil_7', color='green')
plt.title('Temperatura Promedio')
plt.xlabel('Fecha')
plt.ylabel('Temperatura Promedio')
plt.legend()
plt.grid(True)
plt.show()

# Guardar el DataFrame final en un archivo CSV sin incluir el índice
data.to_csv('temperaturas_final.csv', index=True)

# Confirmación de guardado
print("\n¡Datos procesados guardados exitosamente!")

