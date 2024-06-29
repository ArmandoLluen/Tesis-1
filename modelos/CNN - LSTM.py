import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Cargar los datos desde el archivo CSV
df = pd.read_csv('temperaturas_final.csv', parse_dates=['fecha'], index_col='fecha')

# Normalizar los datos
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Función para generar secuencias de tiempo
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data)-seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Definir la longitud de la secuencia de tiempo
seq_length = 10  # ajustar según sea necesario

# Crear secuencias de tiempo para el entrenamiento del modelo
X, y = create_sequences(scaled_data, seq_length)

# Dividir los datos en conjuntos de entrenamiento y prueba
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Ajustar las dimensiones de entrada para LSTM convolucional
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Definir el modelo LSTM convolucional
model = Sequential([
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(seq_length, 1)),
    LSTM(50, activation='relu'),
    Dense(1)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mse')

# Resumen del modelo
model.summary()

# Entrenar el modelo
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Hacer predicciones
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invertir la normalización para obtener predicciones en la escala original
train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1)).flatten()
y_train = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1)).flatten()
y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Calcular métricas de rendimiento
train_rmse = np.sqrt(mean_squared_error(y_train, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predict))
train_mae = mean_absolute_error(y_train, train_predict)
test_mae = mean_absolute_error(y_test, test_predict)

print(f'Train RMSE: {train_rmse:.2f}, Train MAE: {train_mae:.2f}')
print(f'Test RMSE: {test_rmse:.2f}, Test MAE: {test_mae:.2f}')

# Visualización de resultados
plt.figure(figsize=(14, 7))
plt.plot(df.index[-len(test_predict):], y_test, marker='.', label='True')
plt.plot(df.index[-len(test_predict):], test_predict, 'r', marker='.', label='Predicted')
plt.title('Predicciones de temperatura usando LSTM convolucional')
plt.xlabel('Fecha')
plt.ylabel('Temperatura')
plt.legend()
plt.show()

# Definir temperaturas normales estacionales
temperaturas_normales = {
    '01': 22, '02': 23.5, '03': 24,  # Verano (diciembre a marzo)
    '04': 22, '05': 21,  # Primavera y Otoño
    '06': 20, '07': 19, '08': 18, '09': 18,  # Invierno (junio a septiembre)
    '10': 20, '11': 21, '12': 21   # Primavera y Otoño
}

# Calcular la media climatológica estacional usando los datos de entrenamiento
media_climatologica_estacional = df.index.map(lambda d: temperaturas_normales[d.strftime('%m')])

# Calcular las anomalías estacionales en los datos de prueba y predicciones
df_pred = pd.DataFrame({'fecha': df.index[-len(test_predict):], 'tsm_pred': test_predict})
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
