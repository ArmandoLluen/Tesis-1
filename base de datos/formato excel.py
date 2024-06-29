import pandas as pd

# Cargar el archivo Excel
archivo_excel = 'Base_datos_Temperatura_1Ene_31Dic20.xlsx'
print("Cargando el archivo Excel...")
datos = pd.read_excel(archivo_excel)
print("Archivo cargado.")

# Mostrar los datos
print(datos.head())

# Mostrar los nombres de las columnas
print("Nombres de las columnas:")
print(datos.columns)

# Eliminar filas con valores nulos
print("Eliminando filas con valores nulos...")
datos = datos.dropna()
print("Filas con valores nulos eliminadas.")

# Definir las latitudes de la costa norte del Perú
latitud_min = -6.0
latitud_max = -3.5

# Filtrar las coordenadas que están dentro de la región de la costa norte del Perú
print("Filtrando coordenadas de la costa norte del Perú...")
datos_costa_norte = datos[(datos['lat'] >= latitud_min) & (datos['lat'] <= latitud_max)]
print("Coordenadas filtradas.")

# Mostrar los datos filtrados
print(datos_costa_norte.head())

# Extraer las fechas del DataFrame filtrado
print("Extrayendo fechas...")
fechas = datos_costa_norte.columns[2:]  # Suponiendo que las primeras dos columnas son lat y long
print(fechas)

# Calcular el promedio por fecha
print("Calculando el promedio por fecha...")
promedio_por_fecha = datos_costa_norte[fechas].mean(axis=0)  # Calcular el promedio solo para las fechas
print(promedio_por_fecha)

# Crear un rango de fechas desde enero hasta diciembre de 2018
fechas_datatime = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')

# Crear un DataFrame con las fechas y temperaturas promedio
print("Creando DataFrame final...")
df_promedio = pd.DataFrame({
    'fecha': fechas_datatime,
    'temperatura_promedio': promedio_por_fecha.values  # Usar los valores directamente
})
print(df_promedio.head())

# Guardar el DataFrame en un archivo CSV sin incluir el índice
print("Guardando DataFrame en archivo CSV...")
df_promedio.to_csv('temperaturas_promedio_2020.csv', index=False)
print("Archivo CSV guardado.")