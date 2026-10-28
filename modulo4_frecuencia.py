"""
=============================================================================
MÓDULO 4: FILTRADO EN FRECUENCIA (FFT2) Y ACENTUADO ESPACIAL (SIN OPENCV)
=============================================================================
Fundamento Teórico:
- Dominio de la Frecuencia (Transformada 2D de Fourier):
  * Paso 1: F(u,v) = FFT2(f(x,y)). La componente continua DC (0 Hz / brillo promedio)
    nace en la esquina superior izquierda [0, 0].
  * Paso 2: fftshift. Traslada diagonalmente los cuadrantes para situar la frecuencia DC
    en el centro geométrico [M/2, N/2].
  * Paso 3: Multiplicación por la máscara circular H(u,v) con radio de corte D0:
    - Pasa-Bajos Ideal: H(u,v)=1 en el centro (D <= D0) y 0 afuera (atenúa ruido, desenfoca bordes).
    - Pasa-Altos Ideal: H(u,v)=0 en el centro (D <= D0) y 1 afuera (anula fondo, deja solo bordes).
  * Paso 4: ifftshift. Devuelve los cuadrantes a las esquinas para que la IFFT2 interprete
    el origen en [0, 0] sin modulación de signo (-1)^(x+y).
  * Paso 5: IFFT2 y Magnitud Real: Retorno al dominio espacial continuo g(x,y) = |IFFT2(G)|.
- Filtros de Acentuado Espacial:
  * Laplaciano Puro Centro 8: Suma de pesos = 0. En zonas homogéneas suma cero (fondo negro, solo aristas).
  * High-Boost Centro 9: Suma de pesos = 1 (A*Orig + Laplaciano). Restaura el brillo y color base.
=============================================================================
"""

import numpy as np


def acotar_pixel_uint8(valor_pixel):
    """Acota cualquier valor numérico estrictamente al rango entero sin signo [0, 255]."""
    valor_entero = int(round(valor_pixel))
    return max(0, min(255, valor_entero))


def agregar_relleno_ceros(matriz_imagen, relleno_filas, relleno_columnas):
    """Agrega un marco perimetral de ceros para no perder dimensiones en el borde."""
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


def aplicar_acentuado_espacial(
    imagen_u8,
    modo_acentuado="high_boost",
    factor_amplificacion_a=1.2
):
    """
    Acentuado espacial:
    - Laplaciano Centro 8: Suma 0, anula zonas uniformes y extrae solo aristas sobre fondo oscuro.
    - High-Boost: Restituye el brillo base preservando color y afilando bordes (A*Orig + Lap).
    Soporta matrices 2D (escala de grises) y 3D (RGB canal por canal).
    """
    if imagen_u8.ndim == 3:
        canales_procesados = [
            aplicar_acentuado_espacial(
                imagen_u8[:, :, canal],
                modo_acentuado=modo_acentuado,
                factor_amplificacion_a=factor_amplificacion_a
            )
            for canal in range(imagen_u8.shape[2])
        ]
        return np.stack(canales_procesados, axis=-1)

    mascara_laplaciana_8vecinos = np.array(
        [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]],
        dtype=np.float64
    )
    imagen_con_relleno = agregar_relleno_ceros(imagen_u8.astype(np.float64), 1, 1)
    alto_imagen, ancho_imagen = imagen_u8.shape
    imagen_acentuada = np.zeros((alto_imagen, ancho_imagen), dtype=np.uint8)

    for pos_fila in range(alto_imagen):
        for pos_columna in range(ancho_imagen):
            region_vecindad = imagen_con_relleno[pos_fila:pos_fila + 3, pos_columna:pos_columna + 3]
            valor_laplaciano = np.sum(region_vecindad * mascara_laplaciana_8vecinos)
            if modo_acentuado == "solo_bordes":
                pixel_resultado = abs(valor_laplaciano)
            else:
                pixel_resultado = (
                    factor_amplificacion_a * float(imagen_u8[pos_fila, pos_columna])
                    + valor_laplaciano
                )
            imagen_acentuada[pos_fila, pos_columna] = acotar_pixel_uint8(pixel_resultado)

    return imagen_acentuada


def filtrar_en_frecuencia_fft2(
    imagen_u8,
    tipo_filtro="pasa_bajos",
    radio_corte_d0=35.0,
    modo_visualizacion_espectro="logaritmico"
):
    """
    Filtrado en Frecuencia:
    1. FFT2 -> 2. fftshift -> 3. H(u,v) circular -> 4. ifftshift -> 5. IFFT2 (magnitud real).
    Permite elegir el modo de visualización del espectro (Logarítmico, Lineal, Fase o Máscara H(u,v)).
    Soporta matrices 2D (escala de grises) y 3D (RGB canal por canal).
    """
    if imagen_u8.ndim == 3:
        canales_reconstruidos = []
        espectro_visualizacion = None
        for canal in range(imagen_u8.shape[2]):
            canal_rec, espectro_canal = filtrar_en_frecuencia_fft2(
                imagen_u8[:, :, canal],
                tipo_filtro=tipo_filtro,
                radio_corte_d0=radio_corte_d0,
                modo_visualizacion_espectro=modo_visualizacion_espectro
            )
            canales_reconstruidos.append(canal_rec)
            if canal == 0:
                espectro_visualizacion = espectro_canal
        return np.stack(canales_reconstruidos, axis=-1), espectro_visualizacion

    alto_imagen, ancho_imagen = imagen_u8.shape
    transformada_fourier = np.fft.fft2(imagen_u8)
    fourier_centrada_dc = np.fft.fftshift(transformada_fourier)

    # 1. Construcción analítica de la máscara circular de transferencia H(u, v)
    centro_filas_u, centro_columnas_v = alto_imagen // 2, ancho_imagen // 2
    coordenada_u, coordenada_v = np.ogrid[:alto_imagen, :ancho_imagen]
    matriz_distancias_radiales = np.sqrt(
        (coordenada_u - centro_filas_u)**2 + (coordenada_v - centro_columnas_v)**2
    )

    if tipo_filtro == "pasa_bajos":
        mascara_transferencia_h = (matriz_distancias_radiales <= radio_corte_d0).astype(np.float64)
    else:
        mascara_transferencia_h = (matriz_distancias_radiales > radio_corte_d0).astype(np.float64)

    # 2. Modos de Representación del Espectro Fourier
    if modo_visualizacion_espectro in ("mascara_H", "mascara_filtro_H"):
        espectro_visualizacion = mascara_transferencia_h * 255.0
    elif modo_visualizacion_espectro == "lineal":
        espectro_visualizacion = np.abs(fourier_centrada_dc)
    elif modo_visualizacion_espectro == "fase":
        espectro_visualizacion = np.angle(fourier_centrada_dc)
    else:
        espectro_visualizacion = np.log1p(np.abs(fourier_centrada_dc))

    # 3. Filtrado en frecuencia y retorno al dominio espacial con IFFT2
    fourier_filtrada_centrada = fourier_centrada_dc * mascara_transferencia_h
    fourier_filtrada_descentrada = np.fft.ifftshift(fourier_filtrada_centrada)
    imagen_reconstruida_espacio = np.abs(np.fft.ifft2(fourier_filtrada_descentrada))

    return (
        np.clip(np.round(imagen_reconstruida_espacio), 0, 255).astype(np.uint8),
        espectro_visualizacion
    )


# =============================================================================
# ALIAS DE COMPATIBILIDAD (Para no romper código existente)
# =============================================================================
formato_uint8 = acotar_pixel_uint8
agregar_padding_ceros = agregar_relleno_ceros
acentuado_espacial = aplicar_acentuado_espacial
filtrado_frecuencia_fft = filtrar_en_frecuencia_fft2
