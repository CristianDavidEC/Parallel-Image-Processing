import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count


def apply_filter_section(args):
    pixels, start, end, width, kernel = args
    kernel_to_exe = np.array(kernel)
    section_edges = np.zeros_like(pixels[start:end, :])

    dimension_slice = 1 if len(kernel) == 3 else 2
    end_for_i = end - start - 1 if len(kernel) == 3 else end - start - 2
    end_for_j = width - 1 if len(kernel) == 3 else width - 2

    for i in range(dimension_slice, end_for_i):
        for j in range(dimension_slice, end_for_j):
            slice_i_neg = i - dimension_slice + start
            slice_i_pos = i + dimension_slice + 1 + start
            slice_j_neg = j - dimension_slice
            slice_j_pos = j + dimension_slice + 1

            gy = np.sum(np.multiply(
                pixels[slice_i_neg:slice_i_pos, slice_j_neg:slice_j_pos], kernel_to_exe))
            section_edges[i, j] = min(255, np.abs(gy))

    return section_edges


def apply_filter_multiprocessing(path_image, kernel):
    image = Image.open(path_image).convert('L')
    pixels = np.array(image)
    height, width = pixels.shape
    num_processes = cpu_count()

    # Divide the image into sections with overlap
    section_height = height // num_processes
    sections = []
    for i in range(num_processes):
        start = i * section_height
        end = (i + 1) * section_height if i != num_processes - 1 else height
        if i != 0:
            start -= 1  # Overlap to cover edges
        sections.append((pixels, start, end, width, kernel))

    # Create a process pool and apply the filter to each section
    with Pool() as pool:
        results = pool.map(apply_filter_section, sections)

    # Combine the results
    edges = np.vstack(results)

    image_out = Image.fromarray(edges)
    name_image = path_image.split('/')[-1].split('.')[0]
    image_path_out = f'./resources/processed_images/{name_image}.jpg'
    image_out.save(image_path_out)
    
    
