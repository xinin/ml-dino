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
#model.save('modelo_entrenado.h5')

# Convierte el modelo a TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Guarda el modelo convertido en un archivo .tflite
with open('modelo_convertido.tflite', 'wb') as f:
    f.write(tflite_model)



interpreter = tf.lite.Interpreter(model_path='modelo_convertido.tflite')

# Cargar el modelo de TensorFlow Lite
interpreter.allocate_tensors()

# Obtener información de entrada y salida del modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preparar los datos de entrada para la predicción
input_shape = input_details[0]['shape']
input_data = np.random.rand(20, 10)[0].astype(np.float32)  # Ejemplo de datos de entrada


input_details = interpreter.get_input_details()
print("Forma de entrada esperada:", input_details[0]['shape'])


print("Forma de entrada dada:", input_data.shape)

input_data = input_data.reshape((1, 10))
print(input_data)
print("Forma de entrada dada:", input_data.shape)


# Asignar los datos de entrada al tensor de entrada del modelo
interpreter.set_tensor(input_details[0]['index'], input_data)

# Realizar la predicción
interpreter.invoke()

# Obtener los resultados de la predicción
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
predicted_label = np.argmax(output_data)  # Obtener la etiqueta predicha

# Imprimir el resultado
print("Etiqueta predicha:", predicted_label)

