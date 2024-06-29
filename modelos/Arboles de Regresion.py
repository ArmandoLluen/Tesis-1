import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np
import matplotlib.pyplot as plt

# Cargar los datos desde el archivo CSV
df = pd.read_csv('temperaturas_final.csv', parse_dates=['fecha'], index_col='fecha')

# Separar los datos en conjuntos de entrenamiento y prueba
train_data = df[df.index.year < 2023]
test_data = df[df.index.year == 2023]

# Preparación de los datos
def prepare_data(df):
    df = df.copy()  # Hacer una copia del DataFrame para evitar SettingWithCopyWarning
    df['dayofyear'] = df.index.dayofyear
    return df[['dayofyear', 'temperatura_promedio']]

train_data_prepared = prepare_data(train_data)
test_data_prepared = prepare_data(test_data)

# Definir las características y las etiquetas
X_train = train_data_prepared[['dayofyear']]
y_train = train_data_prepared['temperatura_promedio']
X_test = test_data_prepared[['dayofyear']]
y_test = test_data_prepared['temperatura_promedio']

# Optimización de Hiperparámetros usando GridSearchCV
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Crear el modelo
tree = DecisionTreeRegressor()

# Buscar los mejores parámetros
grid_search = GridSearchCV(estimator=tree, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Mejor modelo
best_model = grid_search.best_estimator_

# Validación cruzada
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)
print(f"Cross-Validation RMSE Scores: {cv_rmse}")
print(f"Cross-Validation RMSE Mean: {cv_rmse.mean()}")
print(f"Cross-Validation RMSE Std Dev: {cv_rmse.std()}")

# Predicción
y_pred = best_model.predict(X_test)

# Evaluación del modelo en el conjunto de prueba
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Test RMSE: {rmse}")

# Visualización de los resultados
plt.figure(figsize=(14, 7))
plt.plot(test_data.index, y_test, label='Real', color='blue')
plt.plot(test_data.index, y_pred, label='Predicción', color='red')
plt.xlabel('Fecha')
plt.ylabel('Temperatura Promedio')
plt.title('Predicción de Temperatura Promedio para 2023 (Árbol de Decisión Optimizado)')
plt.legend()
plt.show()