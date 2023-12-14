#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define STB_IMAGE_IMPLEMENTATION //Librerias para cargar imagen
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION //libreria para guardar imagenes.
#include "stb_image_write.h"

void aplicar_filtro_bordes(unsigned char *input, unsigned char *output, int width, int height, int channels) {
    int kernel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int kernel_y[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float gx = 0, gy = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int p = input[(y + ky) * width + (x + kx)];
                    gx += p * kernel_x[ky + 1][kx + 1];
                    gy += p * kernel_y[ky + 1][kx + 1];
                }
            }
            int magnitude = (int)sqrt(gx * gx + gy * gy);
            magnitude = magnitude > 255 ? 255 : magnitude;
            output[y * width + x] = (unsigned char)magnitude;
        }
    }
}

int main() {
    int width, height, channels;
    unsigned char *img = stbi_load("img.jpg", &width, &height, &channels, 1);
    if (img == NULL) {
        printf("Error al cargar la imagen\n");
        return 1;
    }

    unsigned char *output_img = malloc(width * height * sizeof(unsigned char));
    if (output_img == NULL) {
        printf("No se pudo asignar memoria para la imagen de salida\n");
        stbi_image_free(img);
        return 1;
    }

    clock_t start, end;
    double cpu_time_used;

    start = clock();
    aplicar_filtro_bordes(img, output_img, width, height, channels);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    stbi_write_jpg("imagen_con_bordes.jpg", width, height, 1, output_img, 100);

    printf("Tiempo de ejecución: %f segundos\n", cpu_time_used);

    stbi_image_free(img);
    free(output_img);

    return 0;
}