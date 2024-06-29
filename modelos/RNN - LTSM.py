import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.layers import Dropout

# Cargar los datos
data = pd.read_csv('temperaturas_final.csv', parse_dates=['fecha'], index_col='fecha')

# Filtrar datos entre 2017 y 2022 para entrenamiento
train_data = data['2017-01-01':'2022-12-31']
test_data = data['2023-01-01':'2023-12-31']

# Normalización de los datos
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data[['temperatura_promedio']])
scaled_test_data = scaler.transform(test_data[['temperatura_promedio']])

# Crear secuencias de entrenamiento
def create_sequences(data, sequence_length):
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        label = data[i + sequence_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

sequence_length = 30  # Ejemplo de 30 días de secuencia
X_train, y_train = create_sequences(scaled_train_data, sequence_length)

# Crear el modelo LSTM con Dropout
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)))  # Primera capa LSTM
model.add(Dropout(0.2))  # Añadir Dropout para regularización
model.add(LSTM(50, return_sequences=False))  # Segunda capa LSTM
model.add(Dropout(0.2))  # Añadir Dropout para regularización
model.add(Dense(1))  # Capa densa final

model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_train, y_train, epochs=50, batch_size=32)  # Aumentar el número de épocas

# Crear secuencias de prueba
X_test, y_test = create_sequences(scaled_test_data, sequence_length)

# Predicción con el modelo
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Invertir la normalización de y_test para comparación
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calcular las métricas de error
mse = mean_squared_error(y_test_actual, predictions)
mae = mean_absolute_error(y_test_actual, predictions)
mape = np.mean(np.abs((y_test_actual - predictions) / y_test_actual)) * 100

print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Absolute Percentage Error (MAPE): {mape}%')

# Plotear las predicciones versus los valores reales
plt.figure(figsize=(14, 5))
plt.plot(test_data.index[sequence_length:], y_test_actual, color='blue', label='Temperaturas Reales')
plt.plot(test_data.index[sequence_length:], predictions, color='red', label='Predicciones')
plt.xlabel('Fecha')
plt.ylabel('Temperatura Promedio')
plt.title('Predicción de la Temperatura Promedio para 2023')
plt.legend()
plt.show()

# Definir temperaturas normales estacionales
temperaturas_normales = {
    '01': 22, '02': 23.5, '03': 24,  # Verano (diciembre a marzo)
    '04': 22, '05': 21,  # Primavera y Otoño
    '06': 20, '07': 19, '08': 18, '09': 18,  # Invierno (junio a septiembre)
    '10': 20, '11': 21, '12': 21   # Primavera y Otoño
}


# Calcular las anomalías estacionales en los datos de prueba y predicciones
df_pred = pd.DataFrame({'fecha': test_data.index[sequence_length:], 'tsm_pred': predictions.flatten()})
df_pred['mes'] = df_pred['fecha'].dt.strftime('%m')
df_pred['temp_normal'] = df_pred['mes'].map(temperaturas_normales)
df_pred['anomalía'] = df_pred['tsm_pred'] - df_pred['temp_normal']

# Calcular la media móvil de tres meses de las anomalías
df_pred['media_movil'] = df_pred['anomalía'].rolling(window=3).mean()

# Clasificar según el Índice Costero El Niño (ICEN)
def clasificar_icen(valor):
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
    else:
        return 'Indefinido'

df_pred['categoria_icen'] = df_pred['media_movil'].apply(clasificar_icen)

# Filtrar las categorías relevantes
eventos_significativos = df_pred.dropna(subset=['media_movil'])

# Imprimir resultados
print(eventos_significativos)

# Mapa de categorías para el eje Y
categorias = ['La Niña Fuerte', 'La Niña Moderado', 'La Niña Débil', 'Neutro',
              'El Niño Débil', 'El Niño Moderado', 'El Niño Fuerte', 'El Niño Muy Fuerte']

categoria_map = {categoria: idx for idx, categoria in enumerate(categorias)}

# Línea de tiempo de la evolución de la clasificación ICEN
plt.figure(figsize=(14, 7))
df_pred['categoria_idx'] = df_pred['categoria_icen'].map(categoria_map)
plt.plot(df_pred['fecha'], df_pred['categoria_idx'], color='blue', marker='o', linestyle='-', label='Clasificación ICEN')
plt.yticks(ticks=range(len(categorias)), labels=categorias)
plt.title('Línea de Tiempo de la Evolución de la Clasificación ICEN')
plt.xlabel('Fecha')
plt.ylabel('Categoría ICEN')
plt.legend()
plt.show()
