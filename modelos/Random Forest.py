import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Cargar los datos
df = pd.read_csv('temperaturas_final.csv', index_col='fecha', parse_dates=True)

# Dividir los datos en entrenamiento y prueba
train = df.loc['2017':'2022'].copy()  # Usar .loc para seleccionar rangos de fechas
test = df.loc['2023'].copy()  # Usar .loc para seleccionar el año 2023

# Ingeniería de características: añadir un lag de temperatura
train['lag1_temp'] = train['temperatura_promedio'].shift(1)
test['lag1_temp'] = test['temperatura_promedio'].shift(1)

# Eliminar filas con NaN debido al shift
train.dropna(inplace=True)
test.dropna(inplace=True)

# Preparar datos para el modelo
X_train = train[['lag1_temp']]
y_train = train['temperatura_promedio']

X_test = test[['lag1_temp']]
y_test = test['temperatura_promedio']

# Inicializar y entrenar el modelo Random Forest con GridSearchCV para ajustar hiperparámetros
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

# Predecir con datos de prueba
predictions = best_rf.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Visualización de resultados solo para el año 2023
plt.figure(figsize=(12, 6))
plt.plot(test.index, test['temperatura_promedio'], label='Temperatura Observada', color='blue')
plt.plot(test.index, predictions, label='Predicción', color='red')
plt.title('Predicción de Temperaturas con Random Forest para 2023')
plt.xlabel('Fecha')
plt.ylabel('Temperatura Promedio')
plt.legend()
plt.grid(True)
plt.show()
