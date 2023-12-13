from kernels_proccesor.multiprocess import apply_filter_multiprocessing
from kernels.kernels import KERNELS

for key, kernel in KERNELS.items():
    print(key, kernel)
    path = './image.jpg'
    name_image = path.split('/')[-1].split('.')[0]
    image_edges = apply_filter_multiprocessing(path, kernel)
    image_path_out = f'./resources/processed_images/{name_image}_{key}.jpg'
    print(image_path_out)
    image_edges.save(image_path_out)


# kernel = KERNELS['CLASS_1']
# key = 'CLASS_1'
# path = './flower.jpg'
# name_image = path.split('/')[-1].split('.')[0]
# image_edges = apply_filter_multiprocessing(path, kernel)
# image_path_out = f'./resources/processed_images/{name_image}_{key}.jpg'
# print(image_path_out)
# image_edges.save(image_path_out)