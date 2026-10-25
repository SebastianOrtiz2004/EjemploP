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


def formato_uint8(valor):
    """Acota al rango [0, 255]."""
    val = int(round(valor))
    return max(0, min(255, val))


def agregar_padding_ceros(matriz, pad_f, pad_c):
    """Agrega un marco perimetral de ceros para no perder filas ni columnas."""
    alto, ancho = matriz.shape
    salida = np.zeros((alto + 2 * pad_f, ancho + 2 * pad_c), dtype=matriz.dtype)
    salida[pad_f:pad_f + alto, pad_c:pad_c + ancho] = matriz
    return salida


def calcular_matriz_gaussiana(k, sigma=1.0):
    """Genera analíticamente la máscara Gaussiana continua normalizada a suma = 1.0."""
    kernel = np.zeros((k, k), dtype=np.float64)
    centro = k // 2
    for i in range(k):
        for j in range(k):
            x, y = i - centro, j - centro
            kernel[i, j] = math.exp(-(x**2 + y**2) / (2.0 * (sigma**2)))
    return kernel / np.sum(kernel)


def inyectar_ruido_sal_pimienta(matriz_img, densidad=0.05):
    """Inyecta ruido impulsivo (sal=255 y pimienta=0) vectorizado mediante NumPy puro."""
    if densidad <= 0.0:
        return matriz_img.copy()

    salida = matriz_img.copy()
    forma = salida.shape[:2]
    rnd = np.random.rand(*forma)

    # Densidad dividida equitativamente entre sal y pimienta
    mitad_d = densidad / 2.0
    mascara_pimienta = rnd < mitad_d
    mascara_sal = (rnd >= mitad_d) & (rnd < densidad)

    if salida.ndim == 3:
        salida[mascara_pimienta] = [0, 0, 0]
        salida[mascara_sal] = [255, 255, 255]
    else:
        salida[mascara_pimienta] = 0
        salida[mascara_sal] = 255

    return salida


def convolucion_con_salto_manual(canal_u8, mascara, salto_f=1, salto_c=1, usar_pad=True):
    """
    Convolución espacial punto a punto con submuestreo por saltos (stride).
    Cumple con el Teorema de Muestreo de Nyquist mitigando el aliasing.
    """
    k_f, k_c = mascara.shape
    pad_f = (k_f - 1) // 2 if usar_pad else 0
    pad_c = (k_c - 1) // 2 if usar_pad else 0

    alto, ancho = canal_u8.shape
    if usar_pad:
        mat_trabajo = agregar_padding_ceros(canal_u8.astype(np.float64), pad_f, pad_c)
        lim_f, lim_c = alto, ancho
        alto_out = math.ceil(alto / float(salto_f))
        ancho_out = math.ceil(ancho / float(salto_c))
    else:
        mat_trabajo = canal_u8.astype(np.float64)
        lim_f = alto - 2 * pad_f
        lim_c = ancho - 2 * pad_c
        alto_out = math.ceil(lim_f / float(salto_f))
        ancho_out = math.ceil(lim_c / float(salto_c))

    salida = np.zeros((alto_out, ancho_out), dtype=np.uint8)
    suma_w = float(np.sum(mascara))
    if abs(suma_w) < 1e-6:
        suma_w = 1.0

    idx_f = 0
    for f in range(0, lim_f, salto_f):
        if idx_f >= alto_out:
            break
        idx_c = 0
        for c in range(0, lim_c, salto_c):
            if idx_c >= ancho_out:
                break
            vecindad = mat_trabajo[f:f + k_f, c:c + k_c]
            acumulador = np.sum(vecindad * mascara)
            salida[idx_f, idx_c] = formato_uint8(acumulador / suma_w)
            idx_c += 1
        idx_f += 1
        
    return salida


def filtro_mediana_manual(canal_u8, k=3, salto_f=1, salto_c=1):
    """Filtro no lineal: ordena la vecindad espacial y extrae el elemento central."""
    pad = (k - 1) // 2
    mat_pad = agregar_padding_ceros(canal_u8, pad, pad)
    alto, ancho = canal_u8.shape

    alto_out = math.ceil(alto / float(salto_f))
    ancho_out = math.ceil(ancho / float(salto_c))
    salida = np.zeros((alto_out, ancho_out), dtype=np.uint8)

    idx_f = 0
    for f in range(0, alto, salto_f):
        if idx_f >= alto_out:
            break
        idx_c = 0
        for c in range(0, ancho, salto_c):
            if idx_c >= ancho_out:
                break
            vecindad = mat_pad[f:f + k, c:c + k].flatten()
            vecindad.sort()
            salida[idx_f, idx_c] = vecindad[len(vecindad) // 2]
            idx_c += 1
        idx_f += 1
    return salida
