import numpy as np
from sklearn.neural_network import MLPClassifier

# Paso 1: Definir los datos de entrada y salida
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5], [0.25, 0.75], [0.75, 0.25], [0.8, 0.8]])
y = np.array([0, 1, 2, 0, 1, 2, 1, 0])

# Paso 2: Definir la arquitectura de la red neuronal
clf = MLPClassifier(hidden_layer_sizes=(7,), activation='relu', solver='adam', 
                    alpha=0.0001, batch_size='auto', learning_rate='constant', 
                    learning_rate_init=0.001, power_t=0.5, max_iter=200, 
                    shuffle=True, random_state=None, tol=0.0001, verbose=False, 
                    warm_start=False, momentum=0.9, nesterovs_momentum=True, 
                    early_stopping=False, validation_fraction=0.1, beta_1=0.9, 
                    beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)

# Paso 3: Entrenar el modelo
clf.fit(X, y)

# Paso 4: Predecir la salida para nuevos datos
X_new = np.array([[0.4, 0.4], [0.6, 0.6], [0.9, 0.1]])
y_pred = clf.predict(X_new)

# Paso 5: Obtener las probabilidades de clase para nuevos datos
y_proba = clf.predict_proba(X_new)

# Paso 6: Imprimir los resultados
print("Datos de entrada:")
print(X_new)
print("Salida predecida:")
print(y_pred)
print("Probabilidades de clase:")
print(y_proba)
print(y_proba[0])
print(np.argmax(y_proba[0]))
