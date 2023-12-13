from kernels.kernels import KERNELS
import subprocess
import os

for key, kernel in KERNELS.items():
    print(key, kernel)
    # kernel = KERNELS['SQUARE_3X3']
    # key = 'SQUARE_3X3'
    path = '../image.jpg'

    sub_folder = 'kernels_proccesor'
    abs_path = os.path.abspath(sub_folder)
    # Definir el comando a ejecutar
    comando_mpi = f"mpiexec -n 4 python mpi4py_process.py {path} {key} {kernel}"
    # Ejecutar el comando utilizando subprocess
    subprocess.run(comando_mpi, shell=True, cwd=abs_path)