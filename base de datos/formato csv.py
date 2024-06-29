import pandas as pd

# Cargar el archivo CSV
archivo_csv = 'Base_datos_Temperatura_1Ene_31Dic22.csv'
datos = pd.read_csv(archivo_csv)

# Mostrar los datos
print(datos)

# Mostrar los nombres de las columnas
print("Nombres de las columnas:")
print(datos.columns)

# Eliminar filas con valores nulos
datos = datos.dropna()

# Definir las latitudes de la costa norte del Perú
latitud_min = -6.0
latitud_max = -3.5

# Filtrar las coordenadas que están dentro de la región de la costa norte del Perú
datos_costa_norte = datos[(datos['lat'] >= latitud_min) & (datos['lat'] <= latitud_max)]

print(datos_costa_norte)

# Extraer las fechas del DataFrame filtrado
fechas = datos_costa_norte.columns[2:]  # Suponiendo que las primeras dos columnas son lat y long
print(fechas)

# Calcular el promedio por fecha
promedio_por_fecha = datos_costa_norte[fechas].mean(axis=0)  # Calcular el promedio solo para las fechas
print(promedio_por_fecha)

# Crear un rango de fechas desde enero hasta diciembre de 2019
fechas_datatime = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')

# Crear un DataFrame con las fechas y temperaturas promedio
df_promedio = pd.DataFrame({
    'fecha': fechas_datatime,
    'temperatura_promedio': promedio_por_fecha  # Inicialmente estableceremos los valores como None o NaN
})
print(df_promedio)

# Guardar el DataFrame en un archivo CSV sin incluir el índice
df_promedio.to_csv('temperaturas_promedio_2022.csv', index=False)

