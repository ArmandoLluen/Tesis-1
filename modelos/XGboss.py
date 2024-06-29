import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Cargar los datos del archivo CSV
datos = pd.read_csv('temperaturas_final.csv', index_col='fecha', parse_dates=True)

# Dividir los datos en entrenamiento y prueba
train_data = datos[datos.index < '2023-01-01'].copy()
test_data = datos[(datos.index >= '2023-01-01') & (datos.index < '2024-01-01')].copy()

# Feature Engineering: Agregar nuevas características
train_data.loc[:, 'day_of_year'] = train_data.index.day_of_year
train_data.loc[:, 'month'] = train_data.index.month
train_data.loc[:, 'day'] = train_data.index.day
test_data.loc[:, 'day_of_year'] = test_data.index.day_of_year
test_data.loc[:, 'month'] = test_data.index.month
test_data.loc[:, 'day'] = test_data.index.day

# Definir características (X) y variable objetivo (y)
features = ['day_of_year', 'month', 'day']
X_train = train_data[features]
y_train = train_data['temperatura_promedio']
X_test = test_data[features]
y_test = test_data['temperatura_promedio']

# Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definir el modelo
model = xgb.XGBRegressor(objective='reg:squarederror')

# Definir los hiperparámetros a buscar
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5]
}

# Búsqueda en grid con validación cruzada
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Mejor modelo encontrado por la búsqueda en grid
best_model = grid_search.best_estimator_

# Realizar predicciones en los datos de prueba
predictions = best_model.predict(X_test_scaled)

# Evaluar el rendimiento del modelo
mse = mean_squared_error(y_test, predictions)
print(f'Error cuadrático medio (MSE): {mse}')

# Graficar las predicciones vs los valores reales solo para el año 2023
plt.figure(figsize=(10, 6))
plt.plot(test_data.index, y_test, label='Datos reales de prueba')
plt.plot(test_data.index, predictions, label='Predicciones', linestyle='--')
plt.xlabel('Fecha')
plt.ylabel('Temperatura promedio')
plt.legend()
plt.title('Predicción de temperaturas promedio con XGBoost (2023)')
plt.show()

