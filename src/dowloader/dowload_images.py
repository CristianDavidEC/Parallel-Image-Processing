from dotenv import load_dotenv
from tqdm import tqdm
import os
import requests
import os
import math
import threading
import shutil

load_dotenv('../.env')
# Configuración de la API de Pexels
PEXELS_API_KEY = os.getenv('PEXELS_API_KEY')
MAX_NUMBER_IMAGES = 80


def thread_download_images(query, total_images=1000):
    clean_contents()
    TOTAL_THREADS = 10
    iteration_pages = math.ceil(
        (total_images/TOTAL_THREADS) / MAX_NUMBER_IMAGES)
    threads = []
    for i in range(TOTAL_THREADS):
        thread = threading.Thread(target=dowload_images,
                                  args=(query, total_images //
                                        TOTAL_THREADS, i * iteration_pages),
                                  name=f'Thread {i}')
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


def dowload_images(query, number_of_images, page):
    progress_bar = tqdm(total=number_of_images,
                        desc=f'Descargando imágenes - {threading.current_thread().name}: ')
    count_images = 0
    base_url = 'https://api.pexels.com/v1/search'
    params = {'query': query,
              'per_page': MAX_NUMBER_IMAGES, 'page': page}
    # Encabezados con la clave de la API
    headers = {'Authorization': PEXELS_API_KEY}

    while count_images < number_of_images:
        if count_images % MAX_NUMBER_IMAGES == 0:
            page += 1
            params['page'] = page

        try:
            response = requests.get(base_url, headers=headers, params=params)
        except Exception as e:
            print(f'Error al obtener las imágenes: {e}')
            break

        if response.status_code == 200:
            for photo in response.json()['photos']:
                img_url = photo['src']['tiny']
                img_id = photo['id']
                img_name = f'../resources/images/{query}_{img_id}.jpg'

                # Descargar la imagen
                img_data = requests.get(img_url).content
                with open(img_name, 'wb') as img_file:
                    img_file.write(img_data)
                count_images += 1
                progress_bar.update(1)
                if count_images == number_of_images:
                    break
        else:
            print(
                f"Error al obtener las imágenes de la página {page} - {threading.current_thread().name} - {response.status_code}")


def clean_contents():
    path_folder = '../resources/images'
    try:
        shutil.rmtree(path_folder)
        print(f"Contenido eliminado correctamente.")
    except Exception as e:
        print(f"No se pudo eliminar el contenido: {e}")

    
    nueva_carpeta = os.path.join(path_folder)
    os.makedirs(nueva_carpeta)
    print(f"Nueva carpeta creada en {path_folder}")

if __name__ == '__main__':
    thread_download_images('dog', 1000)
