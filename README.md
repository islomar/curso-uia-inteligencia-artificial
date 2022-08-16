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
    - Desviación estándar
        - La desviación estándar y la varianza, indican la desviación con respecto a la media de una población o muestra
    - Mediana: valor que ocupa el lugar central de todos los datos cuando éstos están ordenados de menor a mayor.
    - Moda: Valor que aparece con más frecuencia


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
