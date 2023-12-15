# Paralelización en el Procesamiento de Imágenes

## Introducción
En el campo del procesamiento de imágenes, la optimización de la velocidad es esencial para mejorar la eficiencia y reducir los tiempos de ejecución de algoritmos. Este informe aborda la implementación y comparación de diferentes técnicas de paralelización de procesos para la aplicación de filtros en imágenes.

## Metodología
### Descripción del Problema
El estudio se centra en la aplicación de filtros a un conjunto extenso de imágenes, evaluando el rendimiento de tres técnicas de paralelización: MPI, CUDA y Multiprocessing.

### Objetivos del Estudio
- Evaluar los tiempos de procesamiento al aplicar diferentes filtros mediante técnicas de paralelización.
- Calcular la aceleración lograda por cada método en comparación con el procesamiento secuencial.

### Procedimientos Aplicados
- Preparación de Datos: Selección y recopilación de un conjunto diverso de imágenes.
- Implementación de Filtros: Desarrollo de algoritmos adaptados a cada método de paralelización.
- Ejecución de Pruebas: Procesamiento de imágenes utilizando cada técnica de paralelización.
- Análisis de Resultados: Evaluación comparativa de tiempos y eficiencia de cada método.

### Consideraciones y Validación
Evaluación y análisis de los resultados obtenidos con identificación de posibles mejoras para futuras optimizaciones.

## Resultados
- Descarga paralela de imágenes: Mejora significativa en el tiempo de descarga en comparación con el método secuencial.
- Ejecución de procesos en paralelo: Significativa mejora en tiempos

En Windows:
```bash
python -m venv nombre_del_entorno
nombre_del_entorno\Scripts\activate
```
Mac o Linux

```bash
python3 -m venv nombre_del_entorno
source nombre_del_entorno/bin/activate
```

```bash
pip install -r requirements.txt
```
