import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
from PIL import Image
from kernels.kernels import KERNELS

cuda.init()


def apply_cuda(image_path, kernel):
    image = Image.open(image_path).convert('L')  # Convertir a escala de grises
    image_np = np.array(image).astype(np.float32)
    height, width = image_np.shape
    size_kernel = len(kernel)
    kernel = np.array(kernel).astype(np.float32)

    # Crear arrays en GPU
    image_gpu = cuda.mem_alloc(image_np.nbytes)
    kernel_gpu = cuda.mem_alloc(kernel.nbytes)
    cuda.memcpy_htod(image_gpu, image_np)
    cuda.memcpy_htod(kernel_gpu, kernel)

    output = np.zeros(
        (image_np.shape[0], image_np.shape[1])).astype(np.float32)
    output_cpu = cuda.mem_alloc(output.nbytes)

    threads_per_block = (32, 32, 1)
    blocks_per_grid = (int(np.ceil(width / threads_per_block[0])),
                       int(np.ceil(height / threads_per_block[1])))
    
    mod = SourceModule("""
//cuda
        __global__ void convolution(float *image, float *kernel, float *result, int height, int width, int size_k) {
            int idX = blockIdx.x * blockDim.x + threadIdx.x;
            int idY = blockIdx.y * blockDim.y + threadIdx.y;

            // center
            int c = size_k / 2;

            // calcular la posición central de la máscara (start)
            int startY = idY - c;
            int startX = idX - c;         
  
            float temp = 0.0f;
            for (int y = 0; y < size_k; ++y) {
                for (int x = 0; x < size_k; ++x) {
                    if ((startY + y) >= 0 && (startY + y) < height && (startX + x >= 0 && (startX + x) < width)) {
                        temp += image[(startY + y) * width + (startX + x)] * kernel[y * size_k + x];
                    }
                }
            }

            result[idY * width + idX ] = temp;
        }
//!cuda
    """)

    function_convolution = mod.get_function("convolution")

    # Llamar al kernel CUDA
    function_convolution(image_gpu, kernel_gpu, output_cpu, np.int32(height), np.int32(
        width), np.int32(size_kernel), block=threads_per_block, grid=blocks_per_grid)

    cuda.memcpy_dtoh(output, output_cpu)

    name_image = image_path.split('/')[-1].split('.')[0]
    image_path_out = f'./resources/processed_images/{name_image}.jpg'
    image = np.uint8(output)
    img = Image.fromarray(image)
    img.save(image_path_out)
