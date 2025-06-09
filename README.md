[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/MKRVUeWW)
# Clasificación de Dígitos MNIST con Redes Neuronales Convolucionales
Integrantes: Cristian Hernandez, Samuel Blanchar, Diego Suarez

### Descripción de la tarea
Este proyecto demuestra la implementación y análisis de una Red Neuronal Convolucional (CNN) pre-entrenada para la clasificación de dígitos MNIST. El cuaderno incluye varios ejercicios para comprender y visualizar el comportamiento del modelo, incluyendo visualización de kernels, operaciones de convolución y predicciones en imágenes personalizadas.

### Características
- Carga y uso de un modelo MNIST pre-entrenado
- Evaluación del modelo en datos de prueba
- Visualización y análisis de kernels
- Visualización de operaciones de convolución
- Capacidades de predicción en imágenes personalizadas

### Requisitos
- Python 3.9+
- TensorFlow
- NumPy
- Matplotlib
- Requests

### Uso
1. Ejecutar las celdas del cuaderno en secuencia
2. El modelo se descargará automáticamente desde Hugging Face
3. Seguir los ejercicios en el cuaderno para:
   - Visualizar los kernels del modelo
   - Analizar las operaciones de convolución
   - Realizar predicciones en imágenes personalizadas

### Ejercicios Incluidos
# 1. Visualización de al menos 10 kernels del modelo pre-entrenado
   Se implementa el siguiente código:

#Importar librerías
from tensorflow import keras
load_model = keras.models.load_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import requests

#PASO 1: Descargar el modelo preentrenado
url = "https://huggingface.co/spaces/ayaanzaveri/mnist/resolve/main/mnist-model.h5"
r = requests.get(url)
with open("mnist_model.h5", "wb") as f:
    f.write(r.content)
print("Modelo guardado correctamente.")

#PASO 2: Cargar datos de prueba y el modelo
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

model = load_model("mnist_model.h5")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Evaluar precisión del modelo
loss, accuracy = model.evaluate(x_test, y_test)
print("Precisión en el conjunto de prueba:", accuracy)

#PASO 3: Probar una imagen individual
idx = 0
img = x_test[idx]
plt.imshow(img.squeeze(), cmap='gray')
plt.title(f"Etiqueta real: {y_test[idx]}")
plt.axis('off')
plt.show()

prediction = model.predict(np.expand_dims(img, axis=0))
print("Predicción del modelo:", np.argmax(prediction))

# Visualización de 10 kernels (filtros) de la primera capa convolucional
kernels = model.layers[0].get_weights()[0]
print(f"Shape de los kernels: {kernels.shape}")

fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for i in range(10):
    kernel = kernels[:, :, 0, i]
    axes[i].imshow(kernel, cmap='gray')
    axes[i].axis('off')
    axes[i].set_title(f'Filtro {i+1}')
plt.show() 
# 2. Visualización de las salidas de la convolución con los kernels
Se implementa el siguiente código:
#Importar librerías
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import requests

#PASO 1: Descargar el modelo preentrenado
url = "https://huggingface.co/spaces/ayaanzaveri/mnist/resolve/main/mnist-model.h5"
r = requests.get(url)
with open("mnist_model.h5", "wb") as f:
    f.write(r.content)
print("Modelo guardado correctamente.")

#PASO 2: Cargar datos de prueba y el modelo
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

model = load_model("mnist_model.h5")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Evaluar precisión del modelo
loss, accuracy = model.evaluate(x_test, y_test)
print("Precisión en el conjunto de prueba:", accuracy)

#PASO 3: Probar una imagen individual
idx = 1
img = x_test[idx]
plt.imshow(img.squeeze(), cmap='gray')
plt.title(f"Etiqueta real: {y_test[idx]}")
plt.axis('off')
plt.show()

prediction = model.predict(np.expand_dims(img, axis=0))
print("Predicción del modelo:", np.argmax(prediction))

# kernels (filtros) de la primera capa convolucional
kernels = model.layers[0].get_weights()[0]
print(f"Shape de los kernels: {kernels.shape}")

fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for i in range(10):
    kernel = kernels[:, :, 0, i]
    axes[i].imshow(kernel, cmap='gray')
    axes[i].axis('off')
    axes[i].set_title(f'Filtro {i+1}')
plt.show()

# Se obtiene la primera capa convolucional
from tensorflow.keras.models import Model

#Creamos un modelo intermedio desde la entrada hasta la primera capa convolucional

intermediate_model = Model(inputs=model.layers[0].input, outputs=model.layers[0].output)

#Aplicar la imagen a la primera capa (reshape para que tenga forma [1,28,28,1])
feature_maps = intermediate_model.predict(np.expand_dims(img, axis=0))

# Visualizar los primeros 10 mapas de activación
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))
for i in range(10):
    ax = plt.subplot(1, 10, i + 1)
    plt.imshow(feature_maps[0, :, :, i], cmap='gray')
    plt.title(f"Mapa {i+1}")
    plt.axis('off')
plt.tight_layout()
plt.show()
# 3. Predicciones en al menos 10 imágenes propias
Se implementa el siguiente código:
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
load_model = tf.keras.models.load_model
import tensorflow as tf
image = tf.keras.preprocessing.image

import cv2

# Cargar el modelo previamente entrenado
model = load_model("mnist_model.h5")

# Lista con los nombres de las imágenes
nombres_imagenes = [
    "0a.png", "1.png", "2a.png", "3a.png", "4.png", "5a.png",
    "6.png", "7a.png", "8.png", "9.png"
]

#Procesar y predecir cada imagen
for nombre in nombres_imagenes:
    # Leer la imagen en escala de grises
    img = cv2.imread(nombre, cv2.IMREAD_GRAYSCALE)

    # Verificar que se haya leído correctamente
    if img is None:
        print(f"No se pudo cargar la imagen: {nombre}")
        continue

    # Redimensionar a 28x28
    img = cv2.resize(img, (28, 28))

    # Invertir colores si el fondo es blanco y el dígito negro
    if np.mean(img) > 127:
        img = 255 - img

    # Normalizar valores y dar forma al input (1, 28, 28, 1)
    img = img / 255.0
    img_input = img.reshape(1, 28, 28, 1)

    # Hacer predicción
    pred = model.predict(img_input)
    pred_clase = np.argmax(pred)

    # Mostrar resultados
    print(f"{nombre} → Predicción: {pred_clase}")

    # Mostrar la imagen
    plt.imshow(img, cmap="gray")
    plt.title(f"{nombre} → Predicción: {pred_clase}")
    plt.axis("off")
    plt.show()
# 4. Repetición del análisis de convolución con imágenes personalizadas
   Se implementa el siguiente código:
   #Importar librerías
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import requests

#PASO 1: Descargar el modelo preentrenado
url = "https://huggingface.co/spaces/ayaanzaveri/mnist/resolve/main/mnist-model.h5"
r = requests.get(url)
with open("mnist_model.h5", "wb") as f:
    f.write(r.content)
print("Modelo guardado correctamente.")

#PASO 2: Cargar datos de prueba y el modelo
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

model = load_model("mnist_model.h5")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Evaluar precisión del modelo
loss, accuracy = model.evaluate(x_test, y_test)
print("Precisión en el conjunto de prueba:", accuracy)
#PASO 3: Probar una imagen individual
from PIL import Image

#Cargar imagen, convertir a escala de grises, redimensionar a 28x28
img = Image.open("4.png").convert("L").resize((28, 28))

#Convertir a array, normalizar a [0, 1]
img = np.array(img).astype("float32") / 255.0

#Invertir colores si el fondo está blanco y el número es negro
img = 1.0 - img

#Redimensionar a (28, 28, 1)
img = img.reshape((28, 28, 1))

# Mostrar la imagen
plt.imshow(img.squeeze(), cmap='gray')
plt.title("Imagen invertida: .png")
plt.axis('off')
plt.show()

# Predicción
prediction = model.predict(np.expand_dims(img, axis=0))
print("Predicción del modelo:", np.argmax(prediction))

# Visualizar 10 kernels de la primera capa convolucional
kernels = model.layers[0].get_weights()[0]
print(f"Shape de los kernels: {kernels.shape}")

fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for i in range(10):
    kernel = kernels[:, :, 0, i]
    axes[i].imshow(kernel, cmap='gray')
    axes[i].axis('off')
    axes[i].set_title(f'Filtro {i+1}')
plt.show()

#Se obtiene la primera capa convolucional
from tensorflow.keras.models import Model

#Creamos un modelo intermedio desde la entrada hasta la primera capa convolucional
intermediate_model = Model(inputs=model.layers[0].input, outputs=model.layers[0].output)

#Aplicar la imagen a la primera capa (reshape para que tenga forma [1,28,28,1])
feature_maps = intermediate_model.predict(np.expand_dims(img, axis=0))

# Visualizar los primeros 10 mapas de activación
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))
for i in range(10):
    ax = plt.subplot(1, 10, i + 1)
    plt.imshow(feature_maps[0, :, :, i], cmap='gray')
    plt.title(f"Mapa {i+1}")
    plt.axis('off')
plt.tight_layout()
plt.show()
