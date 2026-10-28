"""
=============================================================================
MÓDULO 3: FILTRADO ESPACIAL, PADDING Y REDUCCIÓN POR SALTO (STRIDE) (SIN OPENCV)
=============================================================================
Fundamento Teórico:
- Dominio del Espacio: Opera directamente sobre las coordenadas físicas (x, y).
- Convolución Espacial: Multiplicación punto a punto de cada celda de la máscara
  por su vecino correspondiente, sumatoria total y normalización (división entre suma de pesos).
  * Sin normalizar: Si la suma de pesos es > 1, la imagen se satura a blanco (255).
- Orden Impar (3x3, 5x5...): Garantiza la existencia de un elemento central geométrico único.
- Padding de Ceros: Añade (K-1)/2 filas y columnas para no perder (N-2)x(M-2) bordes.
- Submuestreo por Saltos (Stride): Reduce la tasa de muestreo. Según Nyquist-Shannon (fs >= 2*fmax),
  la convolución previa actúa como filtro anti-aliasing indispensable antes de descartar filas/columnas.
- Filtros Lineales vs No Lineales:
  * Lineales: Gaussiano (analítico según varianza sigma^2) y Media uniforme (puros 1s).
  * No Lineales: Mediana y Moda (sin pesos, ordenamiento local en vecindad).
=============================================================================
"""

import math
import numpy as np


def acotar_pixel_uint8(valor_pixel):
    """Acota cualquier valor numérico estrictamente al rango entero sin signo [0, 255]."""
    valor_entero = int(round(valor_pixel))
    return max(0, min(255, valor_entero))


def agregar_relleno_ceros(matriz_imagen, relleno_filas, relleno_columnas):
    """Agrega un marco perimetral de ceros para no perder información en los bordes."""
    alto_original, ancho_original = matriz_imagen.shape
    matriz_con_relleno = np.zeros(
        (alto_original + 2 * relleno_filas, ancho_original + 2 * relleno_columnas),
        dtype=matriz_imagen.dtype
    )
    matriz_con_relleno[
        relleno_filas:relleno_filas + alto_original,
        relleno_columnas:relleno_columnas + ancho_original
    ] = matriz_imagen
    return matriz_con_relleno


def generar_mascara_gaussiana(tamano_kernel, desviacion_sigma=1.0):
    """
    Genera analíticamente la máscara Gaussiana continua normalizada para que
    la suma de sus coeficientes sea exactamente 1.0 (evitando alteración de brillo medio).
    """
    mascara_gaussiana = np.zeros((tamano_kernel, tamano_kernel), dtype=np.float64)
    centro_kernel = tamano_kernel // 2
    for fila_k in range(tamano_kernel):
        for columna_k in range(tamano_kernel):
            distancia_x = fila_k - centro_kernel
            distancia_y = columna_k - centro_kernel
            mascara_gaussiana[fila_k, columna_k] = math.exp(
                -(distancia_x**2 + distancia_y**2) / (2.0 * (desviacion_sigma**2))
            )
    return mascara_gaussiana / np.sum(mascara_gaussiana)


def aplicar_ruido_sal_y_pimienta(imagen_entrada, probabilidad_ruido=0.05):
    """
    Inyecta ruido impulsivo bipolar (pimienta = 0 y sal = 255)
    de forma vectorizada mediante NumPy puro.
    """
    if probabilidad_ruido <= 0.0:
        return imagen_entrada.copy()

    imagen_ruidosa = imagen_entrada.copy()
    dimensiones = imagen_ruidosa.shape[:2]
    matriz_aleatoria = np.random.rand(*dimensiones)

    # Densidad dividida equitativamente entre sal y pimienta
    umbral_pimienta = probabilidad_ruido / 2.0
    posiciones_pimienta = matriz_aleatoria < umbral_pimienta
    posiciones_sal = (matriz_aleatoria >= umbral_pimienta) & (matriz_aleatoria < probabilidad_ruido)

    if imagen_ruidosa.ndim == 3:
        imagen_ruidosa[posiciones_pimienta] = [0, 0, 0]
        imagen_ruidosa[posiciones_sal] = [255, 255, 255]
    else:
        imagen_ruidosa[posiciones_pimienta] = 0
        imagen_ruidosa[posiciones_sal] = 255

    return imagen_ruidosa


def convolucionar_con_salto_espacial(
    canal_imagen_u8,
    mascara_filtro,
    salto_filas=1,
    salto_columnas=1,
    aplicar_relleno=True
):
    """
    Convolución espacial punto a punto con submuestreo por saltos (stride).
    Cumple con el Teorema de Muestreo de Nyquist mitigando el aliasing.
    """
    filas_kernel, columnas_kernel = mascara_filtro.shape
    relleno_filas = (filas_kernel - 1) // 2 if aplicar_relleno else 0
    relleno_columnas = (columnas_kernel - 1) // 2 if aplicar_relleno else 0

    alto_original, ancho_original = canal_imagen_u8.shape
    if aplicar_relleno:
        matriz_operacion = agregar_relleno_ceros(
            canal_imagen_u8.astype(np.float64),
            relleno_filas,
            relleno_columnas
        )
        limite_filas = alto_original
        limite_columnas = ancho_original
        alto_reducido = math.ceil(alto_original / float(salto_filas))
        ancho_reducido = math.ceil(ancho_original / float(salto_columnas))
    else:
        matriz_operacion = canal_imagen_u8.astype(np.float64)
        limite_filas = alto_original - 2 * relleno_filas
        limite_columnas = ancho_original - 2 * relleno_columnas
        alto_reducido = math.ceil(limite_filas / float(salto_filas))
        ancho_reducido = math.ceil(limite_columnas / float(salto_columnas))

    imagen_filtrada = np.zeros((alto_reducido, ancho_reducido), dtype=np.uint8)
    suma_pesos_mascara = float(np.sum(mascara_filtro))
    if abs(suma_pesos_mascara) < 1e-6:
        suma_pesos_mascara = 1.0

    indice_fila_salida = 0
    for pos_fila in range(0, limite_filas, salto_filas):
        if indice_fila_salida >= alto_reducido:
            break
        indice_columna_salida = 0
        for pos_columna in range(0, limite_columnas, salto_columnas):
            if indice_columna_salida >= ancho_reducido:
                break
            region_vecindad = matriz_operacion[
                pos_fila:pos_fila + filas_kernel,
                pos_columna:pos_columna + columnas_kernel
            ]
            suma_ponderada = np.sum(region_vecindad * mascara_filtro)
            imagen_filtrada[indice_fila_salida, indice_columna_salida] = acotar_pixel_uint8(
                suma_ponderada / suma_pesos_mascara
            )
            indice_columna_salida += 1
        indice_fila_salida += 1

    return imagen_filtrada


def aplicar_filtro_mediana_con_salto(
    canal_imagen_u8,
    tamano_kernel=3,
    salto_filas=1,
    salto_columnas=1
):
    """Filtro no lineal: ordena la vecindad espacial y extrae el elemento central (mediana)."""
    relleno_borde = (tamano_kernel - 1) // 2
    matriz_con_relleno = agregar_relleno_ceros(canal_imagen_u8, relleno_borde, relleno_borde)
    alto_original, ancho_original = canal_imagen_u8.shape

    alto_reducido = math.ceil(alto_original / float(salto_filas))
    ancho_reducido = math.ceil(ancho_original / float(salto_columnas))
    imagen_filtrada = np.zeros((alto_reducido, ancho_reducido), dtype=np.uint8)

    indice_fila_salida = 0
    for pos_fila in range(0, alto_original, salto_filas):
        if indice_fila_salida >= alto_reducido:
            break
        indice_columna_salida = 0
        for pos_columna in range(0, ancho_original, salto_columnas):
            if indice_columna_salida >= ancho_reducido:
                break
            vector_vecindad = matriz_con_relleno[
                pos_fila:pos_fila + tamano_kernel,
                pos_columna:pos_columna + tamano_kernel
            ].flatten()
            vector_vecindad.sort()
            imagen_filtrada[indice_fila_salida, indice_columna_salida] = vector_vecindad[
                len(vector_vecindad) // 2
            ]
            indice_columna_salida += 1
        indice_fila_salida += 1

    return imagen_filtrada


# =============================================================================
# ALIAS DE COMPATIBILIDAD (Para no romper código existente)
# =============================================================================
formato_uint8 = acotar_pixel_uint8
agregar_padding_ceros = agregar_relleno_ceros
calcular_matriz_gaussiana = generar_mascara_gaussiana
inyectar_ruido_sal_pimienta = aplicar_ruido_sal_y_pimienta
convolucion_con_salto_manual = convolucionar_con_salto_espacial
filtro_mediana_manual = aplicar_filtro_mediana_con_salto
