import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Definir los datos de entrenamiento
train_data = np.random.rand(100, 10)  # Genera una matriz de 100 filas y 10 columnas con valores aleatorios entre 0 y 1
train_labels = np.random.randint(2, size=(100, 1))  # Genera etiquetas binarias aleatorias (0 o 1) para cada fila

# Crear el modelo de red neuronal
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=10))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_data, train_labels, epochs=10, batch_size=32)

# Guardar el modelo entrenado
model.save('modelo_entrenado.h5')

# Cargar el modelo entrenado
loaded_model = tf.keras.models.load_model('modelo_entrenado.h5')

# Ejecutar predicciones con el modelo cargado
test_data = np.random.rand(20, 10)  # Genera una matriz de 20 filas y 10 columnas con valores aleatorios entre 0 y 1
predictions = loaded_model.predict(test_data)

# Imprimir las predicciones
print(predictions)
