# INTRODUCCIÓN PRÁCTICA A LA INTELIGENCIA ARTIFICIAL Y AL DEEP LEARNING (B002)
- Universidad Internacional de Andalucia
- https://www.unia.es/es/oferta-academica/oferta-baeza/item/introduccion-practica-a-la-inteligencia-artifical-y-al-deep-learning
- Del 16 al 19 de agosto de 2022

## Día 1: Introducción al curso. Ciencia de datos y Deep learning
- https://classroom.google.com
- [Perceptron](https://www.simplilearn.com/tutorials/deep-learning-tutorial/perceptron)
- [Backpropagation](https://towardsdatascience.com/understanding-backpropagation-abcc509ca9d0)
- DL Big Bang = DNNs + GPUs + Big Data
    - GPU: gran capacidad de computación en paralelo
    - [DNN: Deep Neural Network](https://www.bmc.com/blogs/deep-neural-network/)
- En 1995, se abandonaron las técnicas basadas en ANN a favor de las técnicas de aprendizaje estadístico (SVM)
    - Cibernética --> Perceptrón --> Algoritmo back-propagation --> Deep Learning
- **Aprendizaje automático**: Búsqueda de relaciones estadísticas entre muestras de un conjunto de datos para reconocer patrones y realizar una acción asociada a cada uno de dichos patrones
- **Aprendizaje**:
    - **Supervisado** (supervised): necesaria la solución al problema para reajustar el model (realimentación)
    - **No supervisado** (unsupervised): no es necesaria la solución al problema para reajustar el modelo. Se utilizan por ejemplo, medidas de similitud para separar clases.
    - **Reforzado** (reinforcement): utiliza algunas soluciones y el grado de bondad de las mismas para determinar "lo buena" que es una solución (recompensa).
- **Tipos de problemas en aprendizaje automático**
    - **Clasificación**: la predicción es categórica (pertenencia a una clase)
    - **Regresión**: la predicción es continua
    - **Generación**: producir nuevas muestras a partir de un modelo generado mediante un proceso de aprendizaje --> Modelos generativos
- **Herramientas más populares** 
    - En Inteligencia Artificial, trabajamos con Tensores
    - Un tensor es una generalización de los conceptos de escalar y matriz. Un tensor puede entenderse como un array multidimensional.
    -Numpy, Tensorflow y PyTorch son librerías para cálculo tensorial
    -Tensorflow, además, proporciona un modelo de programación paralelo, basado en grafos, así como algoritmos de optimización
    - PyTorch proporciona herramientas similares a las de TensorFlow, pero con el mismo modelo de programación que NumPy .

### Ciencia de datos
- **Organización de los datos**: La forma habitual de organizar los datos para comenzar a trabajar con ellos, es en forma de matriz, donde cada fila es una muestra y cada columna una variable.
- **Estadística descriptiva**: 
    - Es el conjunto de técnicas numéricas y gráficas para describir un conjunto de datos sin extraer conclusiones.
    - Muestra vs Población
        - **Población**: universo, conjunto o totalidad de elementos sobre los que se investiga o hace un estudio.
        - **Muestra**: subconjunto de elementos que se seleccionan previamente de una población para realizar un estudio.
    - Media
    - Varianza: Medida de la dispersión de los valores respecto a la media
        - Un algoritmo lo tiene más difícil cuanto mayor sea la varianza.
    - Desviación estándar: raíz cuadrada de la varianza
        - La desviación estándar y la varianza, indican la desviación con respecto a la media de una población o muestra
    - Mediana: valor que ocupa el lugar central de todos los datos cuando éstos están ordenados de menor a mayor.
    - Moda: Valor que aparece con más frecuencia
    - Si la muestra es suficientemente grande, se puede suponer sin equivocarse demasiado que la **distribución es normal** (media=0, varianza=1)
- **Estandarización**
    - Permite comparar puntuaciones de dos sujetos en distintas distribuciones o de un sujeto en distintas variables
    - Es el número de desviaciones típicas que una medida se desvía de su media, de acuerdo a una distribución dada
- **Normalización vs. Estandarización**
    - **Normalización**: básicamente, consiste en modificar o adaptar la escala de los datos, con el fin de facilitar la convergencia de los algoritmos de aprendizaje
    - **Estandarización**: consiste en expresar una variable como el número de desviaciones típicas que la separan de la media. De esta forma, se **unifica** la escala de todas las variables.
    - Dos formas de estandarizar variables
        - Z-score: Suponiendo una distribución normal
            - Usado cuando se conoce la desviación estándar de la población y la muestra es mayor de 30
        - t-score: Suponiendo una distribución t de Student (como la distribución normal pero con las colas más largas, para muestras con valores más extremos).
- **Clasificadores lineales**
    - Dadas parejas, ajusta los pesos "w" para que ante una entrada X, la red proporciones la salida correspondiente Y (aprendizaje) -> f(x) = wx + b (f(x)=y, será por ejemplo +/-1)
    - El cálculo del hiperplano de separación óptimo puede formularse como un problema de optimización
    - [Máquinas de soporte vectorial](https://es.wikipedia.org/wiki/M%C3%A1quinas_de_vectores_de_soporte)
    - Más adelante veremos que una red neuronal con una sola capa es un clasificador lineal
- ¿Cómo entrenamos una modelo o red neuronal?
    - Datos de entrenamiento y test tienen que ser diferentes
    - Capacidad de realizar predicciones sobre datos nuevos
- **Aprendizaje y capacidad de generalización**
    - Estimación del error y ajuste de **hiperparámetros**
    - Para evaluar la capacidad de generalización de los modelos, es necesario definir dos (tres) conjuntos de muestras: entrenamiento, validación y prueba. Training set vs Validation set.
    - _Underfitting_ (faltan parámetros, no ajustamos bien) vs _Overfitting_ (sobreajuste, demasiados parámetros, el modelo no funciona bien con datos que no ha visto nunca)
    - **Curva de aprendizaje**
        - En aprendizaje automático, encontrar el **compromiso sesgo-varianza** minimiza el error de generalización.
        - El **sesgo (bias)** es el error cometido por supuestos erróneos. Un sesgo alto puede hacer que el algoritmo pierda las relaciones entre las características y las salidas objetivo (subajuste).
        - La **varianza** es el error producido por la sensibilidad a pequeñas fluctuaciones en el conjunto de entrenamiento. Una varianza alta puede hacer que el algoritmo modele el ruido aleatorio de los datos (overfitting).
    - Mejora de la capacidad de generalización
        - Sesgo alto.
            - Aumentar el número de muestras de entrenamiento
        - Varianza alta:
            - Parada precoz del algoritmo de entrenamiento (early stopping) >> "fastidiar" el aprendizaje
            - Regularización (ej. Dropout)
    - **Matriz de confusión**: 
        - verdadero positivo (TP), falso positivo (Error Tipo I), falso negativo (Error Tipo II), verdadero negativo (TN)
        - Precisión, Especificidad y Sensibilidad
    - El problema del desbalanceo en los datos:
        - Para evitar una mala interpretación de la métrica accuracy, en estos casos es preferible utilizar el accuracy balanceado. Balanced_accuracy = (sensitivity + specificity)/2

### PyTorch
Pytorch es una librería (o lo que se conoce como un framework) que reúne una serie de utilidades para trabajar con cálculo tensorial. Un Tensor es la versión N-dimensional de una matriz, que puede tener un número arbitrario de dimensiones.

El manejo eficiente de los tensores, y su posibilidad de utilizarlos en procesadores paralelos como las tarjetas gráficas (GPUs) es lo que ha catapultado la revolución del deep learning desde el año 2013.

Hay muchas librerías para trabajar con tensores, entre ellas la más conocida: Tensorflow. Sin embargo, pytorch está ganando mucha fuerza en los últimos años, ya que está soportado por Facebook, y es usado activamente en grandes empresas como Uber, Salesforce o Tesla.

Los más familiares con el lenguaje Python para cálculos técnicos y científicos seguramente conozcan numpy. Pytorch está organizado de forma muy similar, pero para igualar el nivel, y comenzar desde cero, vamos a dar nuestros primeros pasos.



## Día 1: Redes neuronales prácticas con PyTorch (I)
- TBD
- https://hub.docker.com/r/pytorch/pytorch
- [Vídeo "Docker + Pytorch" (27 minutos)](https://www.youtube.com/watch?v=ZtHaaWvuZVg)
    - https://juansensio.com/blog/072_pytorch_docker
    - `docker run -it python:3.9-slim python`
    - `docker run python:3.9-slim jupyter notebook`
    - NVIDIA Docker: needed so that Docker has access to the GPUs (which does not happen by default)
        - https://github.com/NVIDIA/nvidia-docker
        - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker
    - [Catálogo de contenedores de NVIDIA para IA](https://catalog.ngc.nvidia.com/containers)
