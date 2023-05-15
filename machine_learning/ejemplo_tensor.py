import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

# Función para aplicar mutaciones a un modelo
def aplicar_mutaciones(modelo, tasa_mutacion):
    for capa in modelo.layers:
        if isinstance(capa, Dense):
            pesos = capa.get_weights()
            nuevos_pesos = []
            for matriz in pesos:
                if matriz.ndim == 2:
                    forma = matriz.shape
                    mutacion = np.random.uniform(low=-0.1, high=0.1, size=forma)
                    mascara = np.random.choice([0, 1], size=forma, p=[1-tasa_mutacion, tasa_mutacion])
                    nuevos_pesos.append(matriz + mutacion * mascara)
                else:
                    nuevos_pesos.append(matriz)
            capa.set_weights(nuevos_pesos)

# Función para realizar un cruce entre dos modelos
def realizar_cruce(modelo1, modelo2):
    nuevo_modelo = keras.models.clone_model(modelo1)
    for capa1, capa2 in zip(modelo1.layers, modelo2.layers):
        if isinstance(capa1, Dense):
            pesos1 = capa1.get_weights()
            pesos2 = capa2.get_weights()
            nuevos_pesos = []
            for matriz1, matriz2 in zip(pesos1, pesos2):
                forma = matriz1.shape
                mascara = np.random.choice([0, 1], size=forma)
                nuevos_pesos.append(matriz1 * mascara + matriz2 * (1 - mascara))
            capa1.set_weights(nuevos_pesos)
    return nuevo_modelo

# Crear una red neuronal para el ejemplo
modelo = Sequential()
modelo.add(Dense(10, input_shape=(7,), activation='relu'))
modelo.add(Dense(10, activation='relu'))
modelo.add(Dense(2, activation='softmax'))
modelo.compile(optimizer='adam', loss='categorical_crossentropy')

# Generar datos de entrenamiento (ejemplo)
datos_entrenamiento = np.random.random((100, 6))
etiquetas_entrenamiento = np.random.randint(0, 2, (100, 2))

# Entrenamiento con mutaciones y cruces dinámicos
tasa_mutacion = 0.1
tasa_cruce = 0.2

for _ in range(10):  # Número de generaciones/iteraciones
    # Mutación en el modelo actual
    aplicar_mutaciones(modelo, tasa_mutacion)

    # Entrenar el modelo mutado
    modelo.fit(datos_entrenamiento, etiquetas_entrenamiento, epochs=1, batch_size=32)

    # Cruce entre el modelo actual y un modelo clonado
    modelo_clonado = keras.models.clone_model(modelo)
    modelo = realizar_cruce(modelo, modelo_clonado)

    # Restablecer pesos para evitar acumulación de mutaciones/cruces
    K.batch_set_value([(w, np.random.random(w.shape)) for w in modelo.weights])

# Utilizar el modelo entrenado para hacer predicciones
predicciones = modelo.predict(datos_prueba)
