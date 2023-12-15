from kernels.kernels import KERNELS
from mpi_exe import mpi_execute
from kernels_proccesor.multiprocess import apply_filter_multiprocessing
from kernels_proccesor.cuda_process import apply_cuda
from tqdm import tqdm
import os
import time
import random as random
import json
import matplotlib.pyplot as plt


def execute_process():
    kernel = KERNELS['SQUARE_5X5']

    process = ['MPI1', 'MPI4', 'MPI8', 'CUDA', 'Multiprocessing']
    dic_process = {}
    for p in process:
        print(f'Procesando con {p}')
        process, time = launch_process(p, kernel)
        dic_process[process] = time

    print(dic_process)
    return dic_process


def launch_process(process, kernel):
    ruta_carpeta = 'resources/images'
    # Recorre todos los archivos en la carpeta
    list_images = os.listdir(ruta_carpeta)
    list_images = random.sample(list_images, 500)
    progress_bar = tqdm(total=len(list_images), desc=f'Procesando Imagenes')
    timer_init = time.time()
    for ruta_archivo in list_images:
        if os.path.isfile(os.path.join(ruta_carpeta, ruta_archivo)):
            apply_kernel(ruta_archivo, kernel,
                         process)
            progress_bar.update(1)

    timer_end = time.time()
    finish_time = timer_end - timer_init
    print(f'Process executed with {process}')
    print(f'Finish time: {finish_time}')

    return (process, finish_time)


def apply_kernel(image, kernel, parallel_computing):
    path = f'./resources/images/{image}'
    if parallel_computing == 'MPI1':
        mpi_execute(f'.{path}', kernel, 1)

    elif parallel_computing == 'MPI4':
        mpi_execute(f'.{path}', kernel, 4)

    elif parallel_computing == 'MPI8':
        mpi_execute(f'.{path}', kernel, 8)

    elif parallel_computing == 'CUDA':
        apply_cuda(path, kernel)

    elif parallel_computing == 'Multiprocessing':
        apply_filter_multiprocessing(path, kernel)


def graph_results():
    ruta_archivo = 'results1000.json'
    with open(ruta_archivo, 'r') as archivo:
        datos_json = json.load(archivo)

    plt.figure(figsize=(10, 6))
    plt.bar(datos_json.keys(), datos_json.values(), color='skyblue')
    plt.title('Tiempos de ejecución de procesos')
    plt.xlabel('Proceso')
    plt.ylabel('Tiempo (segundos)')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Mostrar la gráfica
    plt.show()

    tiempo_secuencial_MPI1 = datos_json['MPI1']
    aceleracion_MPI1 = {proceso: tiempo_secuencial_MPI1 / tiempo_paralelo for proceso, tiempo_paralelo in datos_json.items() if proceso != 'MPI1'}

    # Graficar aceleración de MPI1 respecto a los demás procesos
    plt.figure(figsize=(8, 5))
    plt.bar(aceleracion_MPI1.keys(), aceleracion_MPI1.values(), color='lightgreen')
    plt.axhline(y=1, color='r', linestyle='--', label='Aceleración ideal (1x)')
    plt.title('Aceleración de MPI1 respecto a otros procesos')
    plt.xlabel('Proceso')
    plt.ylabel('Aceleración (MPI1 / Proceso)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Mostrar la gráfica de aceleración
    plt.show()
    


def save_results(dic_results):
    # Escribir el diccionario en un archivo JSON
    nombre_archivo = "results.json"
    with open(nombre_archivo, "w") as archivo:
        json.dump(dic_results, archivo)

    print(f"Diccionario guardado en '{nombre_archivo}'")


if __name__ == '__main__':
    dic_results = execute_process()
    save_results(dic_results)
    graph_results()
