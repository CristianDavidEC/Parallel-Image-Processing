from mpi4py import MPI
from PIL import Image
import numpy as np
import sys
import ast


args = sys.argv[1:]


def format_kernel(kernel):
    complete_kernel = ''.join(kernel)
    final_kernel = np.array(ast.literal_eval(complete_kernel))

    return final_kernel


def apply_filter(fragment_pixel, kernel):
    height, width = fragment_pixel.shape
    kernel_to_exe = np.array(kernel)
    section_edges = np.zeros_like(fragment_pixel)

    dimension_slice = 1 if len(kernel) == 3 else 2

    for i in range(dimension_slice, height - dimension_slice):
        for j in range(dimension_slice, width - dimension_slice):
            slice_i_neg = i - dimension_slice
            slice_i_pos = i + dimension_slice + 1
            slice_j_neg = j - dimension_slice
            slice_j_pos = j + dimension_slice + 1

            processed_area = fragment_pixel[slice_i_neg:slice_i_pos,
                                            slice_j_neg:slice_j_pos]

            gy = np.sum(np.multiply(processed_area, kernel_to_exe))
            section_edges[i, j] = min(255, np.abs(gy))

    return section_edges


def apply_filter_MPI():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    path_image = args[0]
    kernel = args[1:]
    kernel = format_kernel(kernel)

    if rank == 0:
        # Cargar la imagen original
        original_image = Image.open(path_image).convert('L')
        pixels = np.array(original_image)

        # Dividir la imagen en fragmentos para cada proceso
        sendbuf = np.array_split(pixels, size, axis=0)

    else:
        sendbuf = None

    # Scatter los fragmentos de imagen a todos los procesos
    recvbuf = comm.scatter(sendbuf, root=0)

    # Aplicar el filtro Sobel horizontal en paralelo
    result_fragment = apply_filter(recvbuf, kernel)

    # Gather los fragmentos procesados en el proceso 0
    gathered_result = comm.gather(result_fragment, root=0)

    if rank == 0:
        # Reconstruir la imagen final a partir de los fragmentos procesados
        result_image = np.vstack(gathered_result)
        final_image = Image.fromarray(result_image)
        save_image(final_image, path_image)


def save_image(image, path_image):
    name_image = path_image.split('/')[-1].split('.')[0]
    image_path_out = f'../resources/processed_images/{name_image}.jpg'
    image.save(image_path_out)


if __name__ == '__main__':
    apply_filter_MPI()
