---
title: "Diseño Evolutivo de Redes Neuronales con Evolución Diferencial"
author:
    - Alfredo Gutiérrez Alfaro
theme: "metropolis"
fonttheme: "default"
urlcolor: "green"
linkstyle: "bold"
fontsize: 11
aspectratio: 169
section-titles: false
toc: true

header-includes: |
  `\setbeamertemplate{section in toc}[round]`{=latex}
  \setbeamerfont{section number projected}{size=\large}
  \setbeamercolor{section number projected}{bg=red,fg=green} 
  \metroset{sectionpage=none}
  \bibliographystyle{splncs04}
  \usepackage[spanish]{babel}
  \usepackage{subcaption}
  \usepackage{multirow}
  \usepackage{amsfonts}
  \usepackage{bm}
  \usepackage{amsmath}
  \usepackage[linesnumbered,ruled,vlined]{algorithm2e}
  \usepackage{listings}
---

# Introducción
- A la hora de implementar redes neuronales para resolver un problema
requiere de personas expertas para diseñar la topología de la red y
paramétros, entre otros elementos de diseño. [@lopez-vazquez-2019]

- Para resolver esta problemática del diseño, en la literatura se puede
encontrar el uso de metaheuristicas bioinspiradas, como lo es el uso del
Particle Swarm Optimization (PSO), Ant-Colony y la Evolución Diferencial. 
[@garro-2015;@lopez-vazquez-2019;@Alba-Cisneros2020]

# Objetivo
- Implementar y analizar la evolución diferencial para el diseño de redes
neuronales y cómo se desempeñan en las tareas de clasificación.

- Comparar resultados con una red neuronal "tradicional" entrenada con el 
descenso del gradiente

# Redes Neuronales
::: columns

:::: column
Las redes neuronales son un modelo matemático que intentan replicar de cierta
forma a las neuronas biológicas. 

\begin{equation}
  \sigma (\sum_{i=1}^m x_i w_i + b) = \sigma(x^T w + b) = \hat{y}
\end{equation}

::::


:::: column
![Ejemplo red neuronal [@bento-2022]](images/network.png)
::::

::::

# Redes Neuronales
## Funciones de activación

Las funciones de activación introducen no linealidad dentro
de una red neuronal, lo que le permite a la red realizar representaciones
más complejas de los datos.

\begin{table}[]
\begin{tabular}{|l|l|}
\hline
\textbf{Nombre} & \multicolumn{1}{c|}{\textbf{Función}}          \\ \hline
Sigmoid         & $\sigma(x) = \frac{1}{1 + e^{-x}}$             \\ \hline
Tanh            & $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ \\ \hline
Sinusoidal      & $\sin(x)$                                      \\ \hline
Linear          & $f(x) = x$                                     \\ \hline
Hard Limit & $f(x) = \begin{cases} 1 & \text{si } x \geq 0 \\ 0 & \text{si } x < 0 \end{cases}$    \\ \hline
ReLU            & $f(x) = \max(0, x)$                            \\ \hline
Leaky ReLU & $f(x) = \begin{cases} 0.1x & \text{si } x < 0 \\ x & \text{si } x \geq 0 \end{cases}$ \\ \hline
\end{tabular}
\end{table}

# Aprendizaje evolutivo
Los algoritmos evolutivos son un tipo de algoritmos de optimización
heurísticos y aleatorizados, inspirados en la evolución natural. Simulan el proceso de evolución natural considerando dos factores clave: la reproducción variacional y la selección del más apto. [@zhou-2019]

Los estructura básica de la mayoría de algoritmos evolutivos se puede resumir
de la siguiente manera:
\begin{enumerate}
  \item Generar un conjunto inicial de soluciones (llamado población).
  \item Reproducir nuevas soluciones basadas en la población actual, mediante procesos como el cruce y la mutación.
  \item Eliminar las peores soluciones de la población.
  \item Repetir desde el paso 2 hasta que se cumpla algún criterio de parada.
\end{enumerate}

# Evolución diferencial
La evoluación diferencial es un algoritmo evolutivo que sirve para 
resolver problemas continuos. La ED utiliza una estrategia multiparental 
para generar posibles soluciones. El algoritmo se basa en la reproducción
de uno o más individuos, reemplazando a los padres por hijos con mejor aptitud.
[@du-2016]

# Evolución diferencial
## Pseudocódigo
\begin{algorithm}[H]
  \SetAlgoLined
  \SetKwInOut{Input}{Input}
  \SetKwInOut{Output}{Output}
  
  \Input{Number of individuals NP}
  \Output{Optimized solution P}
  
  Generate P = (x1, x2, ..., xNP)\;
  
  \Repeat{stopping condition is satisfied}{
    \For{i = 1 to NP}{
      Compute a mutant vector vi\;
      Create ui by the crossover of vi and xi\;
      
      \If{$f(ui) < f(xi)$}{
        Insert ui into Q\;
      }
      \Else{
        Insert xi into Q\;
      }
    }
    P $\leftarrow$ Q\;
  }
  
  \caption{Differential Evolution (DE)}
\end{algorithm}

# Representación del problema

## Metodología
Diseñar una red neuronal de tres capas con algorimos evolutivos
(evolución diferencial) con un esquema de codificación directo [@garro-2015], 
que tiene en consideración los siguientes elementos:

1. Definir el número de neuronas en la capa oculta
2. Establecer funciones de activación
3. Generar conexiones sinapticas y pesos 

# Representación del problema

## Definir el número de neuronas en la capa oculta

\begin{equation} \label{eq:total}
Q = (M + N) + \frac{N+M}{2}
\end{equation}

\begin{equation} \label{eq:hidden}
H = Q - (M + N)
\end{equation}

\begin{equation} \label{eq:dim}
dim_d = [H * (N + 3)] + [M * (H + 3)]
\end{equation}

# Representación del problema
A continuación una muestra de cómo se representa la codificación para 
el diseño de las redes neuronales:

![Esquema para representar los paramétros [@Alba-Cisneros2020]](images/representation.png)

# Métricas
Para poder evaluar la red y utilizar una función a optimizar dentro de la
evolución diferenciar se hará uso de la exactitud y el error de esta:

\begin{equation}
Exactitud = \frac{VP+VN}{VP+VN+FP+FN}
\end{equation}

\begin{equation}
error = 1 - exactitud
\end{equation}

# Experimentos
Se utilizaron las ecuaciones [\ref{eq:total}, \ref{eq:hidden}, \ref{eq:dim}]
para generar las neuronas (arquitectura) de una red neuronal para dos conjuntos
de datos diferentes: Iris Plant [@misc_iris_53] y Wine [@misc_wine_109].

Posteriormente esta red fue codificadas para ser utilizadas por la evolución
diferencial y así obtener la topología de la red así como sus paramétros para
después evaluar la red con la exactitud.

# Resultados
Aqui pongo gráfica de las redes para ambos datasets

# Resultados
Aquí agrego dos gráficas para comparar la exactitud de la evolución con scikit


# Conclusión
Aqui pongo algo

# Referencias
\bibliographystyle{splncs04}
\bibliography{Gutierrez2023.bib}