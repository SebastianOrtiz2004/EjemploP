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


def formato_uint8(valor):
    """Acota al rango [0, 255]."""
    val = int(round(valor))
    return max(0, min(255, val))


def agregar_padding_ceros(matriz, pad_f, pad_c):
    """Agrega un marco perimetral de ceros."""
    alto, ancho = matriz.shape
    salida = np.zeros((alto + 2 * pad_f, ancho + 2 * pad_c), dtype=matriz.dtype)
    salida[pad_f:pad_f + alto, pad_c:pad_c + ancho] = matriz
    return salida


def acentuado_espacial(matriz_u8, modo="high_boost", factor_a=1.2):
    """
    Acentuado espacial:
    - Laplaciano Centro 8: Suma 0, anula el fondo y deja solo orillas sobre negro.
    - High-Boost: Restituye el brillo base preservando color y afilando aristas (A*Orig + Lap).
    Soporta matrices 2D (escala de grises) y 3D (RGB canal por canal).
    """
    if matriz_u8.ndim == 3:
        canales = [acentuado_espacial(matriz_u8[:, :, c], modo=modo, factor_a=factor_a) for c in range(matriz_u8.shape[2])]
        return np.stack(canales, axis=-1)

    mask_8 = np.array([[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]], dtype=np.float64)
    mat_pad = agregar_padding_ceros(matriz_u8.astype(np.float64), 1, 1)
    alto, ancho = matriz_u8.shape
    salida = np.zeros((alto, ancho), dtype=np.uint8)

    for f in range(alto):
        for c in range(ancho):
            vecindad = mat_pad[f:f + 3, c:c + 3]
            laplaciano = np.sum(vecindad * mask_8)
            if modo == "solo_bordes":
                val = abs(laplaciano)
            else:
                val = factor_a * float(matriz_u8[f, c]) + laplaciano
            salida[f, c] = formato_uint8(val)
    return salida


def filtrado_frecuencia_fft(matriz_u8, tipo="pasa_bajos", d0=35.0, modo_espectro="logaritmico"):
    """
    Filtrado en Frecuencia:
    1. FFT2 -> 2. fftshift -> 3. H(u,v) circular -> 4. ifftshift -> 5. IFFT2 (magnitud real).
    Permite elegir el modo de visualización del espectro (Logarítmico, Lineal, Fase o Máscara H(u,v)).
    Soporta matrices 2D (escala de grises) y 3D (RGB canal por canal).
    """
    if matriz_u8.ndim == 3:
        canales_rec = []
        espectro_vis = None
        for c in range(matriz_u8.shape[2]):
            rec_c, esp_c = filtrado_frecuencia_fft(matriz_u8[:, :, c], tipo=tipo, d0=d0, modo_espectro=modo_espectro)
            canales_rec.append(rec_c)
            if c == 0:
                espectro_vis = esp_c
        return np.stack(canales_rec, axis=-1), espectro_vis

    alto, ancho = matriz_u8.shape
    F = np.fft.fft2(matriz_u8)
    F_shift = np.fft.fftshift(F)

    # 1. Máscara circular H(u, v)
    u_c, v_c = alto // 2, ancho // 2
    u, v = np.ogrid[:alto, :ancho]
    distancias = np.sqrt((u - u_c)**2 + (v - v_c)**2)

    if tipo == "pasa_bajos":
        H = (distancias <= d0).astype(np.float64)  # 1 en centro, 0 afuera
    else:
        H = (distancias > d0).astype(np.float64)   # 0 en centro (bloquea DC), 1 afuera

    # 2. Modos de Espectro
    if modo_espectro == "mascara_H" or modo_espectro == "mascara_filtro_H":
        espectro_vis = H * 255.0
    elif modo_espectro == "lineal":
        espectro_vis = np.abs(F_shift)
    elif modo_espectro == "fase":
        espectro_vis = np.angle(F_shift)
    else:
        espectro_vis = np.log1p(np.abs(F_shift))

    # 3. Multiplicación punto a punto y retorno con IFFT
    G_shift = F_shift * H
    G = np.fft.ifftshift(G_shift)
    img_rec = np.abs(np.fft.ifft2(G))
    return np.clip(np.round(img_rec), 0, 255).astype(np.uint8), espectro_vis
