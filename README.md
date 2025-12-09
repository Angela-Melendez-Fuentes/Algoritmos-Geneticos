# Proyecto 1 - Optimizador de Cortes 2D: Comparativa de Algoritmos (Genético vs. Cuckoo)

Este proyecto implementa y compara dos enfoques para resolver el **Problema de Corte de Material (Cutting Stock)**. 
Logrando acomodar un conjunto de piezas rectangulares en hojas de material de tamaño fijo, **minimizando el desperdicio** y la cantidad de hojas utilizadas.

El proyecto incluye visualización gráfica de los patrones de corte utilizando **matplotlib**.

## Funcionalidad Principal

El sistema toma una lista de piezas requeridas (ancho, alto, cantidad) y las dimensiones del material base. Utiliza algoritmos metaheurísticos para encontrar la mejor disposición posible respetando cortes tipo guillotina (cortes de lado a lado).

### Características
- **Entrada dinámica:** El usuario puede definir dimensiones del plano y agregar múltiples tipos de piezas.
- **Rotación automática:** El algoritmo decide si rotar la pieza 90° para aprovechar mejor el espacio.
- **Visualización:** Genera un gráfico detallado de las hojas, tiras y pilas resultantes.
- **Comparativa:** Implementación de dos lógicas distintas de optimización.
  
## Comparativa de Algoritmos
Este repositorio contiene dos implementaciones distintas para resolver el mismo problema:

### 1. Algoritmo Genético (GA)
Basado en la teoría de la evolución natural.
- **Representación:** El "cromosoma" es una permutación del orden en que se colocan las piezas.
- **Mecanismos:**
    - **Selección:** Torneo (elige los mejores de un subgrupo aleatorio).
    - **Cruce (Crossover):** Cruce por Orden (OX) para mantener la validez de la permutación.
    - **Mutación:** Intercambio (Swap) y movimiento de bloques.
- **Estrategia de Colocación (Heurística):** Utiliza un enfoque **"First Fit" (El primero que encaje)**. Intenta colocar la pieza en la pila actual; si no cabe, intenta en la tira actual; si no, abre una nueva tira o una nueva hoja. Es rápido pero puede dejar huecos.

### 2. Cuckoo Search (CS) - *Búsqueda de Cuco*
Basado en el comportamiento de parasitismo de cría de algunas especies de aves.
- **Representación:** Igual que el GA, optimiza la secuencia de entrada.
- **Mecanismos:**
    - **Vuelos de Lévy:** Utiliza una distribución de probabilidad de "cola pesada" para realizar saltos grandes en el espacio de búsqueda (exploración global agresiva) y saltos pequeños (explotación local).
    - **Abandono de Nidos ($P_a$):** Descarta una fracción de las peores soluciones y las reemplaza con nuevas soluciones aleatorias para evitar mínimos locales.
- **Estrategia de Colocación (Heurística Mejorada):** Utiliza un enfoque **"Best Fit / Scoring"**.
    - A diferencia del GA, este algoritmo **evalúa todas las posiciones posibles** (pilas existentes, tiras existentes, espacios nuevos) y les asigna un puntaje.
    - Prioriza los **"Golden Matches"**: Lugares donde la altura de la pieza coincide exactamente con la altura de la tira, eliminando desperdicio vertical.

### Tabla comparativa: Genético vs. Cuckoo 
| Característica | Algoritmo Genético (GA) | Cuckoo Search (CS) |
| :--- | :--- | :--- |
| **Búsqueda** | Evolutiva (Padres e Hijos) | Nidos y Vuelos de Lévy |
| **Exploración** | Limitada por la tasa de mutación | Alta (Saltos largos de Lévy) |
| **Lógica de Acomodo** | Secuencial (Intenta A -> B -> C) | Basada en Puntuación (Busca el menor desperdicio) |
| **Complejidad** | Media | Alta (Requiere evaluar más posiciones) |
| **Calidad de Solución** | Buena | **Superior** (Tiende a compactar mejor) |

<img width="358" height="370" alt="image" src="https://github.com/user-attachments/assets/dd39a8a2-2ee8-45a7-a81f-aec83d136070" />

<img width="359" height="356" alt="image" src="https://github.com/user-attachments/assets/4b758da0-2e3d-466e-afef-938af1b574a9" />


## Requisitos e Instalación

Este proyecto utiliza Python 3. 
Para garantizar la correcta **visualización de los planos de corte** y el **funcionamiento de los algoritmos numéricos**, asegúrate de tener instaladas las siguientes dependencias:

```bash
pip install matplotlib numpy
```



# Proyecto 2 - Optimizador de Forrado de Libros y Cuadernos

Este es un caso de uso aplicado del algoritmo de optimización. Diseñado específicamente para negocios de papelería o servicios de forrado de libros, este script no solo acomoda rectángulos, sino que **calcula las dimensiones reales de corte** necesarias para forrar un libro basándose en su tamaño estándar y grosor.

## Objetivo del Negocio
Minimizar el desperdicio de papel (Contact, lustre, plástico) al forrar múltiples libros de diferentes tamaños en un rollo o pliego de dimensiones limitadas.

## Características Específicas

### 1. Cálculo de Despliegue (Unfolding)
A diferencia del optimizador genérico, este sistema toma un objeto 3D (un libro cerrado) y calcula su área 2D necesaria para el forrado, incluyendo márgenes automáticos:
- **Desglose:** *Pestaña Izq + Tapa Trasera + Lomo + Tapa Delantera + Pestaña Der*.
- **Márgenes:** Agrega automáticamente 2cm de pestaña por lado para el doblado.
- **Lomo Dinámico:** Permite definir si el encuadernado es de espiral (lomo ~2cm) o personalizado (ej. libros de texto gruesos).

### 2. Catálogo de Tamaños Estándar
Incluye una base de datos interna con medidas comunes de papelería para agilizar la entrada de datos:
| Nombre | Medidas (cm) |
| :--- | :--- |
| **A4 / Letter** | 21 x 29.7 |
| **A5 / Profesional** | 14.8 x 21 |
| **A6 / A7** | Tamaños de bolsillo |
| **Oficio / Folio** | 21.5 x 31.5 |

### 3. Visualización 
El gráfico generado es mucho más detallado. No solo muestra el corte, sino que dibuja las **líneas de doblado**:
- **Gris muy claro:** Área de pestañas (márgenes).
- **Azul/Naranja:** Tapas delantera y trasera.
- **Gris oscuro:** Lomo del libro.

<img width="369" height="392" alt="image" src="https://github.com/user-attachments/assets/ac4137c2-9743-4185-b8b3-5cbfaa374a2b" />


## Lógica del Algoritmo
Utiliza el mismo motor de **Algoritmo Genético**, pero con una capa de abstracción adicional:
1.  **Entrada:** El usuario selecciona "Cuaderno Profesional (A4)".
2.  **Transformación:** El sistema calcula: $Ancho = (21 \times 2) + Lomo + (2 \times 2)$.
3.  **Optimización:** El algoritmo genético busca el mejor acomodo para estas nuevas dimensiones extendidas.


## Requisitos e Instalación

Este proyecto utiliza Python 3. 
Para garantizar la correcta **visualización de los planos de corte** asegúrate de tener instalada la siguiente dependencia:

```bash
pip install matplotlib 
```
