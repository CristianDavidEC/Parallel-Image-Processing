from dotenv import load_dotenv
import os
import requests
import os

load_dotenv('../.env')
# Configuración de la API de Pexels
PEXELS_API_KEY = os.getenv('PEXELS_API_KEY')


def dowload_images(query, number_of_images, page):
    MAX_NUMBER_OF_IMAGES = 80
    count_images = 0
    base_url = 'https://api.pexels.com/v1/search'
    # Parámetros de la solicitud GET
    params = {'query': query,
              'per_page': MAX_NUMBER_OF_IMAGES, 'page': page}
    # Encabezados con la clave de la API
    headers = {'Authorization': PEXELS_API_KEY}

    while count_images <= number_of_images:
        if count_images % MAX_NUMBER_OF_IMAGES == 0:
            page += 1
            params['page'] = page

        response = requests.get(base_url, headers=headers, params=params)

        if response.status_code == 200:
            for photo in response.json()['photos']:
                img_url = photo['src']['tiny']
                img_id = photo['id']
                img_name = f'../resources/images/{SEARCH_QUERY}_{img_id}.jpg'

                # Descargar la imagen
                img_data = requests.get(img_url).content
                with open(img_name, 'wb') as img_file:
                    img_file.write(img_data)
                print(f"Imagen descargada: {img_name}")
                count_images += 1
                if count_images == number_of_images:
                    break
        else:
            print(
                f"Error al obtener las imágenes. Código de estado: {response.status_code}")


if __name__ == '__main__':
    SEARCH_QUERY = input('Ingrese el término de búsqueda: ')
    NUMBER_OF_IMAGES = int(
        input('Ingrese la cantidad de imágenes a descargar: '))
    PAGE = int(input('Ingrese la página a descargar: '))
    dowload_images(SEARCH_QUERY, NUMBER_OF_IMAGES, PAGE)
