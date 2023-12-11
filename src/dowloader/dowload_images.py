from dotenv import load_dotenv
from tqdm import tqdm
import os
import requests
import os

load_dotenv('../.env')
# Configuración de la API de Pexels
PEXELS_API_KEY = os.getenv('PEXELS_API_KEY')


def dowload_images(query, number_of_images, page):
    MAX_NUMBER_IMAGES = 80
    progress_bar = tqdm(total=number_of_images, desc='Descargando imágenes')
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

        response = requests.get(base_url, headers=headers, params=params)

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
                f"Error al obtener las imágenes. Código de estado: {response.status_code}")



