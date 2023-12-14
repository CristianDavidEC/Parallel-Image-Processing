import os
import threading
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random as ramdom
from dowloader.dowload_images import thread_download_images
from dotenv import load_dotenv
from kernels.kernels import KERNELS
from mpi_exe import mpi_execute
from kernels_proccesor.multiprocess import apply_filter_multiprocessing
from kernels_proccesor.cuda_process import apply_cuda
from tqdm import tqdm

def main():
    print('PARALLEL COMPUTING IMAGES')
    kernel = kernel_options()
    kenerl_to_use = KERNELS[kernel]

    print('\n')
    parallel_computing = parallel_computing_options()
    print(f'Parallel computing to use: {parallel_computing}')

    ruta_carpeta = 'resources/images'
    # Recorre todos los archivos en la carpeta
    list_images = os.listdir(ruta_carpeta)
    progress_bar = tqdm(total=len(list_images), desc=f'Procesando Imagenes')
    timer_init = time.time()
    for ruta_archivo in list_images:
        if os.path.isfile(os.path.join(ruta_carpeta, ruta_archivo)):
            apply_kernel(ruta_archivo, kenerl_to_use, parallel_computing)
            progress_bar.update(1)

    timer_end = time.time()

    finish_time = timer_end - timer_init
    print(f'Finish time: {finish_time}')
    print('Finish Execution')


def apply_kernel(image, kernel, parallel_computing):
    path = f'./resources/images/{image}'
    if parallel_computing == 'MPI':
        mpi_execute(f'.{path}', kernel)
    elif parallel_computing == 'CUDA':
        apply_cuda(path, kernel)
    elif parallel_computing == 'Multiprocessing':
        apply_filter_multiprocessing(path, kernel)


def info_dowload():
    print('PARALLEL COMPUTING IMAGES')

    number_images = input('Enter the number of images to process: ')
    number_images = int(number_images)

    theme = input('Enter the theme of the images: ')
    theme = str(theme)

    size = size_options()
    print('\n')
    thread_download_images(theme, size, number_images)
    print('\n')


def size_options():
    options = {
        1: 'tiny',
        2: 'small',
        3: 'medium',
        4: 'large'
    }

    while True:
        print('Select the size of the images: ')
        for key, value in options.items():
            print(f'{key}. {value}')
        option = input('Enter the option: ')
        option = int(option)

        if option in options.keys():
            return options[option]
        else:
            print('Invalid option')


def kernel_options():
    options = {
        1: 'CLASS_1',
        2: 'CLASS_2',
        3: 'CLASS_3',
        4: 'SQUARE_3X3',
        5: 'EDGE_3X3',
        6: 'SQUARE_5X5',
        7: 'EDGE_5X5',
        8: 'SOBLE_VERTICAL',
        9: 'SOBLE_HORIZONTAL',
        10: 'KERNEL_LAPLACE',
        11: 'PREWITT_VERTICAL',
        12: 'PREWITT_HORIZONTAL'
    }

    while True:
        print('Select the kernel to use: ')
        for key, value in options.items():
            print(f'{key}. {value}')
        option = input('Enter the option: ')
        option = int(option)

        if option in options.keys():
            return options[option]
        else:
            print('Invalid option')


def parallel_computing_options():
    options = {
        1: 'MPI',
        2: 'CUDA',
        3: 'Multiprocessing'
    }

    while True:
        print('Select the parallel computing to use: ')
        for key, value in options.items():
            print(f'{key}. {value}')
        option = input('Enter the option: ')
        option = int(option)

        if option in options.keys():
            return options[option]
        else:
            print('Invalid option')

def show_images():
    path_images = './resources/images'
    path_process = './resources/processed_images'
    list_images = os.listdir(path_images)
    list_images = ramdom.sample(list_images, 10)

    for ruta_imagen in list_images:
        img = mpimg.imread(f'{path_images}/{ruta_imagen}')
        img_process = mpimg.imread(f'{path_process}/{ruta_imagen}')
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img)
        axs[0].set_title('Original')
        axs[1].imshow(img_process, cmap='gray')
        axs[1].set_title('Procesada')
        axs[0].axis('off')
        axs[1].axis('off')
        plt.show()


if __name__ == '__main__':
    # th_dw = threading.Thread(target=info_dowload)
    # th_dw.start()
    # th_dw.join()
    # print('\n')
    # time.sleep(3)
    # print('\n')
    # main()

    show_images()
