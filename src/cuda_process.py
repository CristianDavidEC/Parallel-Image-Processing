import numpy as np
from numba import cuda
from PIL import Image
from kernels.kernels import KERNELS


@cuda.jit
def convolution(image, kernel, result):
    # #calcular el id de los hilos para los ejes X (columna) e Y(renglon)
    colId, rowId = cuda.grid(2)
    m = kernel.shape[0]
    height, width = image.shape

    # center
    c = m // 2
    # calcular la posicion central de la mascara (start)
    startRow = rowId - c
    startCol = colId - c

    temp = 0
    # iterar sobre el kernel
    for i in range(m):
        for j in range(m):
            # revisar que no supere el limite de las dimensiones de la imagen
            if int(startRow + i) >= 0 and int(startRow + i) < width:
                if int(startCol + j) >= 0 and int(startCol + j) < height:
                    temp += image[int(startRow + i),
                                 int(startCol + j)] * kernel[i, j]

    result[rowId, colId] = temp


def apply_cuda(image_path, kernel):
    image = Image.open(image_path).convert('L')  # Convertir a escala de grises
    image_np = np.array(image).astype(np.float32)
    height, width = image_np.shape
    size_kernel = len(kernel)

    # Crear arrays en GPU
    image_gpu = cuda.to_device(image_np)
    kernel_gpu = cuda.to_device(np.array(kernel).astype(np.float32))
    output_gpu = cuda.to_device(np.zeros_like(image_np))

    n_threads = 16
    n_blocks = (width + n_threads - 1) // n_threads
    print(f'Blocks: {n_blocks}, Threads: {n_threads}')
    threads_per_block = (n_threads, n_threads)
    blocks_per_grid = (n_blocks, n_blocks)

    # Llamar al kernel CUDA
    convolution[blocks_per_grid, threads_per_block](image_gpu, kernel_gpu, output_gpu)

    res = output_gpu.copy_to_host()

    return res


# Ejemplo de uso
kernel = KERNELS['SOBLE_VERTICAL']
image_path = './flower.jpg'
output_image = apply_cuda(image_path, kernel)

# Guardar la imagen resultante usando PIL
output_image = np.uint8(output_image)
result_image = Image.fromarray(output_image)
result_image.show()  # Mostrar la imagen
result_image.save('resultado_sobel_vertical.jpg')  # Guardar la imagen
