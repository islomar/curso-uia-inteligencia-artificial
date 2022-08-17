# INTRODUCCIÓN PRÁCTICA A LA INTELIGENCIA ARTIFICIAL Y AL DEEP LEARNING (B002)
- Universidad Internacional de Andalucia
- https://www.unia.es/es/oferta-academica/oferta-baeza/item/introduccion-practica-a-la-inteligencia-artifical-y-al-deep-learning
- Del 16 al 19 de agosto de 2022
- Dr. Andrés Ortiz García (Universidad de Málaga)
- Dr. Francisco J. Martínez Murcia (Universidad de Granada)
- Compraron GPU con 48 GB


## Introducción al curso. Ciencia de datos y Deep learning
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
- https://colab.research.google.com/
- Read about the origin of PyTorch and about what a tensor is.
- Pytorch es una librería (o lo que se conoce como un framework) que reúne una serie de utilidades para trabajar con cálculo tensorial. **Un Tensor** es la versión N-dimensional de una matriz, que puede tener un número arbitrario de dimensiones.
- El manejo eficiente de los tensores, y su posibilidad de utilizarlos en procesadores paralelos como las tarjetas gráficas (GPUs) es lo que ha catapultado la revolución del deep learning desde el año 2013.
- Hay muchas librerías para trabajar con tensores, entre ellas la más conocida: Tensorflow. Sin embargo, pytorch está ganando mucha fuerza en los últimos años, ya que está soportado por Facebook, y es usado activamente en grandes empresas como Uber, Salesforce o Tesla.
- Los más familiares con el lenguaje Python para cálculos técnicos y científicos seguramente conozcan numpy. Pytorch está organizado de forma muy similar, pero para igualar el nivel, y comenzar desde cero, vamos a dar nuestros primeros pasos.
- **Atributos de los tensores**: Los tensores son en realidad un "objeto". No sólo contienen los datos en sí (ceros, unos, etc), sino uqe también tienen unas propiedades y unos métodos que es posible visualizar, y que nos van a ayudar mucho en la vida.
- Todas estas pueden ser realizadas en la CPU (procesador del ordenador) o la GPU (procesador gráfico, más rápido para operaciones paralelas). Por defecto, los tensores están creados en la CPU, pero podemos moverlos entre ambas. Como estamos en Colab, si por defecto no tenemos GPU podemos hacerlo en "Entorno de Ejecución> Cambiar tipo de entorno de ejecución", y seleccionar un acelerador por GPU.
- [CUDA](https://developer.nvidia.com/cuda-python)
    - https://www.pcmag.com/encyclopedia/term/cuda
    - Compute Unified Device Architecture
- **Torch.autograd**:
    - Como hemos comentado anteriormente, las redes neuornales están definidas fundamentalmente por tres pasos:
        - Propagación hacia adelante (forward pass): se introduce unos datos en la red y se realizan los cálculos necesarios hasta dar una salida.
        - Cálculo de la pérdida (loss). Se compara la salida de la red con la salida esperada para cuantificar lo "acertado" de nuestra red.
        - Propagación hacia atrás (backward pass o backpropagation). Se calcula el gradiente de la pérdida con respecto a cada entrada y salida de las neuronas, y se ajustan los parámetros
    - Para el último paso, es importante ser capaces de calcular el gradiente, el diferencial de una entrada con respecto a una salida, para poder actualizar los pesos de la red neuronal. Pero, ¿cómo calculamos el gradiente automáticamente?
    - Ahí está el truco del almendruco. torch.autograd lo hace por nosotros.


## Redes neuronales prácticas con PyTorch (I)
- https://hub.docker.com/r/pytorch/pytorch
- [Vídeo "Docker + Pytorch" (27 minutos)](https://www.youtube.com/watch?v=ZtHaaWvuZVg)
    - https://juansensio.com/blog/072_pytorch_docker
    - `docker run -it python:3.9-slim python`
    - `docker run python:3.9-slim jupyter notebook`
    - NVIDIA Docker: needed so that Docker has access to the GPUs (which does not happen by default)
        - https://github.com/NVIDIA/nvidia-docker
        - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker
    - [Catálogo de contenedores de NVIDIA para IA](https://catalog.ngc.nvidia.com/containers)
- [Slides](https://drive.google.com/file/d/1uujG0jZS5Y9Zv8azhIX_7bjz4CP6OyWp/view)
- **¿Qué es una red neuronal?**
    - Una red neuronal artificial (RNA) es un modelo computacional formado por la conexión de unidades que, individualmente, realizan un cálculo sencillo
    - La teoría conexionista, que está detrás de las RNA, unidades simples interconectadas entre sí, pueden resolver problemas complejos
- Una **neurona artificial** implementa un clasificador lineal. Sólo separa clases linealmente ! separables (mediante una recta).
- **Red neuronal básica**: La introducción de una capa de neuronas permite combinar varias rectas (planos) de decisión y así separar clases no linealmente separables.
- Busca calcular los pesos y la constante b.
- El objetivo del aprendizaje es la **generalización**. Memorización vs Generalización.
    - Como después veremos, este ajuste de los pesos se realiza mediante un proceso de optimización de una función objetivo
- Funcionamiento de una red neuronal. Entrenamiento
    - Propagación hacia adelante = Proceso de inferencia --> estimación del error
    - Propagación hacia atrás: ajuste de los pesos para minimizar el error. Una vez detecto un error (e.g. decide que una naranja es una manzana), e van cambiando lo pesos hacia arriba
- Algoritmo backpropagation
    - **Función de pérdida**: calcula la diferencia entre lo que debería salir y lo que ha salido
    - Todas las muestras van acompañadas de su etiqueta. Vector de etiqueta que indica a qué clase pertence (e.g. una manzana va a ser un 0, una naranja un 1).
- **Perceptrón multicapa**
    - Capa de entrada (con tantas neuronas como características tenga) --> Capas ocultas --> Capa de salida (con tantas neuronas como requiera el problema; e.g. para diferenciar entre dos clases, con una neurona sería suficiente. Incluso para una decisión binaria, es mejor tener dos neuronas - una estará totalmente activdada y la otra totalmente desactivada)
- **Función de activación**: hace que la respuesta de una neurona pueda no ser lineal, e.g. Sigmoide, tanh, ReLU, Leaky RELU
- Se comienza con unos pesos aleatorios y luego es van actualizando en baes a los errores (se requiere el gradiente del error con respecto a cada peso así como una tasa de aprendizaje).
- **Proceso de optimización**: Algoritmo backpropagation. Implementación del **Algoritmo de Descenso de Gradiente Estocástico (SGD)**
    - Existen múltiples algoritmos de optimización en Python.
    - El **algoritmo de gradiente descendente** es un método de optimización para encontrar el mínimo local de una función diferenciable.
    - El objetivo es determinar los **parámetros** que minimizan una función de coste
    - Requiere que la función a optimizar sea **convexa** (una función real es convexa en un intervalo (a,b), si la cuerda que une dos puntos cualesquiera en el grafo de la función queda por encima de la función.)
    - Es el método de optimización más usado en la actualidad en Deep Learning.
    - Estamos bien si estamos en una pérdida del 1% (0.01)
- **fc**:
    - capa "Full Connected". Todas conectadas con todas.
    - Se define la conexión entre capas
- **¿Para qué sirven las capas ocultas?**
    - Las capas ocultas proporcionan la capacidad discriminante de una red neuronal
    - **Al incrementar el número de neuronas de una capa**, añadimos parámetros a esa capa, lo que permitirá un mejor ajuste a los datos de entrenamiento. Sin embargo, **reducimos el poder de generalización de la red --> overfitting**
    - **Al añadir capas**, incrementamos la complejidad dimensional que la red es capaz de aprender: **modificamos la forma del hiperplano** de separación.
    - Regla para calcular el óptimo número de capas y neuronas por capa: NO HAY --> Ensayo y error
- **Carga de datos: ¿ Qué ocurre si el tensor X es muy grande y no cabe en memoria (problema muy frecuente en Deep Learning!)?**
    - Se pasa en batches
    - En este caso, tendremos que ir cargando los datos poco a poco  entrenamiento batch
    - Un batch es un subconjunto de datos que SÍ cabe en memoria
    - La idea es calcular la salida y estimar los gradientes del error para cada batch
    - El tamaño del batch puede afectar a la estabilidad numérica del algoritmo de optimización (gradiente descendente). La capacidad de generalización depende del tamaño del batch!. Es, por tanto, **uno de los hiperparámetros más importantes a optimizar**
    - Opinión: usar potencia de 2 para el tamaño del batch.
- **Mejora de la capacidad de generalización. Regularización**
    - La regularización consiste en aplicar una penalización a la función de coste durante el proceso de optimización para evitar el sobreajuste a los datos de entrenamiento.
    - "Fastidiar" el proceso de aprendizaje
    - **Dropout**: Se desactivan aleatoriamente un porcentaje predeterminado de neuronas durante el entrenamiento
        - Evita que las neuronas memoricen parte de la entrada
    - **Early stopping**

### Práctica 2. Redes Neuronales Básicas
- https://colab.research.google.com/
- https://colab.research.google.com/drive/1hGCFeBoEMf-ezK_b7Nqops8gmC_Pv8Yt?hl=es#scrollTo=RrVKNWW_V52U
- 41x41x3=5043 píxeles --> Necesito en la capa de entrada 5043 neuronas
    - Imagen pequeña de 41x41. En color (RGB = 3 pixels, 3 canales de color).
    - Capa de salida: 2 neuronas ("No aceituna (0)" y "Aceituna (1)") --> one-hot encoding
    - Una capa oculta de 1024 neuronas (un solo plano).
    - Es habitual ir disminuyendo el número de neuronas en las capas ocultas conforme se acercan a la salida.
    - La primera capa no tiene por qué ser potencia de 2. Pero el resto mejor sí (opinión personal, creo).
- `x.view(-1, self.input_neurons)`: deja la primera dimensión igual (el número de aceitunas) y el reto lo conviertes a una única dimensión (una fila por imagen, con todos los píxels).
- [Función softmax](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html)
- DataLoader: extraerá la i

## Redes neuronales prácticas con PyTorch (II)
- Deep Learning: más moderno, se usan millones de parámetros.
- Red neuronal: La introducción de una capa de neuronas permite combinar varias regiones (áreas) de decisión y así separar clases no linealmente separables
- Fases:
    1. Datos
        -  Recopilar datos
        -  Ordenarlos (curación de datos)
        -  Procesamiento
    1. Modelo
        - Elección de modelo
        - Construcción de la red
    3. Entrenamiento
        - Bucle de entrenamiento
        - Forward pass
        - Loss
        - Backward pass
    5. Evaluación
        - Estimar la capacidad de generalización
        - Generalmente: conjunto de tests
- **Epoch**: iteración de entrenamiento
- **Redes convolucionales**:
    - intentan mantener la relación que existe entre los pixeles (e.g. en una imagen).
    - Cada pixel lo multiplica por una matriz de kernels o filtros (que representan alguna característica, e.g. bordes en una imagen).
    - Esperamos que el algoritmos extraiga la característica que sea óptima
- **Parámetros de la convolución**
    - Tamaño del kernel (k)
    - Padding: relleno con valores (p)
    - Stride: paso (s) --> distance between two consecutive positions of the kernel (cuánto salta)
    - Dilatación (d): "dilation", cuánto se expande la entrada
    - Todas las neuronas comparten y aplican el mismo peso/kernel/filtro
- [Convolution arithmetic tutorial](https://theano-pymc.readthedocs.io/en/latest/tutorial/conv_arithmetic.html)
- [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285)
- Convolución transpuesta: la inversa de la convolución.
- Arquitecturas convolucionales
    - ALEXNET (y LeNet): lo petó
        - **Pooling**: de un grupo de NxN, selecciona el máximo o promedio. Se hace para reducir el tamaño de entrada.
    - VGG16
- **lr**: learning rate, cuánto modificamos los pesos en cada iteración
- **softmax** va a devolver valores entre 0 y 1, viene bien para tener probabilidades.
- **Adam**: modelo de optimización más sencillito
- Los principales problemas de memoria suelen venir de las capas FC (Fully Connected).
- Otro indicador acerca de las predicciones de nuestro modelo es la "matriz de confusión" (lo encontraréis como confussion matrix).
- **Transfer learning**
    - Transferir el aprendizaje en un dominio (ej. Imagenet) a otro nuevo dominio con un tamaño muestral menor.
    - Redes que han sido entrenados con muchísimas imágenes.
    - ¿Cómo?
        a. Carga red pre-entrenada
        b. Modifica capas a conveniencia (no convolucionales)
        c. Re-entrena con LR baja
- Capas convolucionales: extrae características generales
- Modelo/red "squeezenet", consume poca memoria, pensado para móviles.
- Es buena práctica usar el "early stopper"
- Para leer: https://distill.pub/2019/activation-atlas/


## Aprendizaje no supervisado
- No es necesaria la solución al problema para reajustar el modelo. Se utilizan por ejemplo, medidas de similitud para separar clases.
- Algoritmos de Inteligencia Artificial para identificar patrones en conjuntos de datos no etiquetados y sin conocimiento previo.
- Son útiles para encontrar características que pueden ser útiles para la categorización (agrupamiento)
- Es más fácil obtener datos no etiquetados, dado que el etiquetado de las muestras suele ser un proceso manual (muestra a muestra).
- **Principales aplicaciones**
    - **Agrupamiento de datos** (clustering) de acuerdo a su similitud (medida de similitud)
    - **Detección de anomalías**: desviaciones con respecto a un comportamiento definido como normal
    - **Compresión**: todos los puntos en un cluster pasan a ser representados por el centróide de dicho cluster
    - **Modelos de variables latentes**: compresión, reducción de la dimensionalidad, eliminación de ruído
- **Principales inconvenientes**
    - No se puede tener certeza acerca de la precisión, dado que no disponemos de etiquetas
    - Requiere de una interpretación a posteriori para identificar los grupos
- **Tipos de clustering**
    - Hard-clustering: categorización absoluta, binaria.
    - Soft-clustering: categorización probabilística.
- [Diagrama de Voronoi](https://asignatura.us.es/fgcitig/contenidos/gctem3ma.htm)
- **Hard-Clustering. Algoritmo K-medias**
    - k: número de grupos distintos
- ¿Cómo podemos saber el número óptimo de clusters (k)?
    - Método Elbow
    - https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
    - https://en.wikipedia.org/wiki/Elbow_method_(clustering)
- **Soft-Clustering. Algoritmo Fuzzy C-medias**
- **Proyección lineal. Análisis de Componentes Principales (PCA)**
    - PCA es una técnica estadística que permite describir un conjunto de datos en términos de nuevas variables no correlacionadas.
    - Estas nuevas variables, llamadas componentes principales, explican la varianza de las variables originales, de forma que aquellas componentes que expliquen mayor varianza indicarían las direcciones de máxima variación.
    - Proyectando sobre las PCs:
        - Eliminamos ruido
        - Podemos reducir la dimensionalidad de los datos




## Sistemas de Recomendación
- TBD


## Preguntas
- ¿En el mundo real usáis PyTorch? Sí, tanto en local como en su propio hierro.
- ¿Algo similar a los tests automatizados en desarrollo de SW?
- ¿Por qué se aplana la curva de aprendizaje cuando se aumenta el número de capas?
