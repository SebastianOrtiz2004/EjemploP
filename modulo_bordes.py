"""
=============================================================================
MÓDULO 6: DETECCIÓN DE BORDES, GRADIENTES, LOG, FREI-CHEN Y RELAJACIÓN (SIN OPENCV)
=============================================================================
Fundamento Teórico:
- Operadores de 1er Orden: Roberts (2x2), Prewitt (3x3), Sobel (3x3).
  Sobel descompone en derivación [-1, 0, 1] y suavizado gaussiano [1, 2, 1]^T.
- Operadores Direccionales: Kirsch (4 máscaras a 0°, 45°, 90°, 135° con máxima respuesta).
- 2da Derivada: Laplaciano de una Gaussiana (LoG / DoG) y cruce por cero (zero-crossing).
- Base Ortogonal de Frei-Chen: 9 máscaras (4 orillas, 4 líneas, 1 uniforme).
- Relajación: Refuerzo iterativo contextual de contornos contiguos y supresión de ruido.
=============================================================================
"""

import math
import numpy as np


def convolucion_2d_reflect(matriz, kernel):
    """Convolución espacial 2D manual con padding reflejado en bordes (NumPy puro)."""
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    pad_img = np.pad(matriz.astype(np.float64), ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    H, W = matriz.shape
    salida = np.zeros((H, W), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            salida += kernel[i, j] * pad_img[i:i + H, j:j + W]
    return salida


# =============================================================================
# 1. OPERADORES DE GRADIENTE DE PRIMER ORDEN
# =============================================================================

def operador_roberts_2x2(gris_u8):
    """Operador de Roberts: diferencias cruzadas diagonales en ventana 2x2."""
    H, W = gris_u8.shape
    pad = np.pad(gris_u8.astype(np.float64), ((0, 1), (0, 1)), mode='edge')
    gx = pad[:H, :W] - pad[1:H + 1, 1:W + 1]
    gy = pad[:H, 1:W + 1] - pad[1:H + 1, :W]
    mag = np.abs(gx) + np.abs(gy)
    return gx, gy, mag


def operador_prewitt_3x3(gris_u8):
    """Operador de Prewitt: derivadas ortogonales 3x3 con promediado simple."""
    kx = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
    ky = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    gx = convolucion_2d_reflect(gris_u8, kx)
    gy = convolucion_2d_reflect(gris_u8, ky)
    mag = np.abs(gx) + np.abs(gy)
    ang = np.degrees(np.arctan2(gy, gx + 1e-10)) % 180
    return gx, gy, mag, ang


def operador_sobel_3x3(gris_u8):
    """Operador de Sobel: derivación combinada con suavizado Gaussiano [1, 2, 1]."""
    kx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    ky = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    gx = convolucion_2d_reflect(gris_u8, kx)
    gy = convolucion_2d_reflect(gris_u8, ky)
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.degrees(np.arctan2(gy, gx + 1e-10)) % 180
    return gx, gy, mag, ang


# =============================================================================
# 2. OPERADORES DIRECCIONALES DE KIRSCH (4 ORIENTACIONES)
# =============================================================================

def operador_kirsch_4(gris_u8):
    """
    Operadores direccionales de Kirsch a 0°, 45°, 90° y 135°.
    Calcula la máxima respuesta direccional en cada píxel.
    """
    k0 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
    k45 = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]], dtype=np.float64)
    k90 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    k135 = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]], dtype=np.float64)

    r0 = convolucion_2d_reflect(gris_u8, k0)
    r45 = convolucion_2d_reflect(gris_u8, k45)
    r90 = convolucion_2d_reflect(gris_u8, k90)
    r135 = convolucion_2d_reflect(gris_u8, k135)

    stack = np.stack([np.abs(r0), np.abs(r45), np.abs(r90), np.abs(r135)], axis=0)
    mag_max = np.max(stack, axis=0)
    dir_idx = np.argmax(stack, axis=0)
    angulos = np.array([0, 45, 90, 135])[dir_idx]

    return r0, r45, r90, r135, mag_max, angulos


# =============================================================================
# 3. LAPLACIANO DE UNA GAUSSIANA (LOG / DOG) Y CRUCE POR CERO
# =============================================================================

def operador_log_dog(gris_u8, sigma1=0.5, sigma2=0.8, umbral_cruce=3.0):
    """
    Aproximación DoG (Diferencia de Gaussianas) al Laplaciano de una Gaussiana.
    Extrae los bordes detectando los Cruces por Cero (Zero-Crossings) con cambio de signo.
    """
    def kernel_gaussiano(s):
        k = int(2 * math.ceil(3 * s) + 1)
        if k % 2 == 0:
            k += 1
        c = k // 2
        y, x = np.ogrid[-c:k - c, -c:k - c]
        h = np.exp(-(x**2 + y**2) / (2.0 * s * s))
        return h / np.sum(h)

    g1 = kernel_gaussiano(sigma1)
    g2 = kernel_gaussiano(sigma2)
    k_max = max(g1.shape[0], g2.shape[0])

    if g1.shape[0] < k_max:
        pad_size = (k_max - g1.shape[0]) // 2
        g1 = np.pad(g1, pad_size, mode='constant')
    elif g2.shape[0] < k_max:
        pad_size = (k_max - g2.shape[0]) // 2
        g2 = np.pad(g2, pad_size, mode='constant')

    dog_kernel = g1 - g2
    dog = convolucion_2d_reflect(gris_u8, dog_kernel)

    # Detección de Cruces por Cero (Zero-Crossing) en 4 direcciones ortogonales y diagonales
    H, W = dog.shape
    cruces = np.zeros((H, W), dtype=np.uint8)

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if (
                (dog[i - 1, j] * dog[i + 1, j] < 0 and abs(dog[i - 1, j] - dog[i + 1, j]) > umbral_cruce) or
                (dog[i, j - 1] * dog[i, j + 1] < 0 and abs(dog[i, j - 1] - dog[i, j + 1]) > umbral_cruce) or
                (dog[i - 1, j - 1] * dog[i + 1, j + 1] < 0 and abs(dog[i - 1, j - 1] - dog[i + 1, j + 1]) > umbral_cruce) or
                (dog[i - 1, j + 1] * dog[i + 1, j - 1] < 0 and abs(dog[i - 1, j + 1] - dog[i + 1, j - 1]) > umbral_cruce)
            ):
                cruces[i, j] = 255

    return dog, cruces


# =============================================================================
# 4. MÁSCARAS ORTOGONALES DE FREI-CHEN (9 MÁSCARAS)
# =============================================================================

def operador_frei_chen_9(gris_u8):
    """
    Base vectorial ortogonal de 9 máscaras de Frei-Chen:
    - 4 de Orillas (W1 a W4)
    - 4 de Líneas (W5 a W8)
    - 1 Promedio Uniforme (W9)
    Retorna la proyección de energía en el subespacio de orillas y de líneas.
    """
    s2 = math.sqrt(2.0)
    # 4 Orillas
    w1 = (1.0 / (2.0 * s2)) * np.array([[1, s2, 1], [0, 0, 0], [-1, -s2, -1]])
    w2 = (1.0 / (2.0 * s2)) * np.array([[1, 0, -1], [s2, 0, -s2], [1, 0, -1]])
    w3 = (1.0 / (2.0 * s2)) * np.array([[0, -1, s2], [1, 0, -1], [-s2, 1, 0]])
    w4 = (1.0 / (2.0 * s2)) * np.array([[s2, -1, 0], [-1, 0, 1], [0, 1, -s2]])
    # 4 Líneas
    w5 = 0.5 * np.array([[0, 1, 0], [-1, 0, -1], [0, 1, 0]])
    w6 = 0.5 * np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]])
    w7 = (1.0 / 6.0) * np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
    w8 = (1.0 / 6.0) * np.array([[-2, 1, -2], [1, 4, 1], [-2, 1, -2]])
    # 1 Uniforme
    w9 = (1.0 / 3.0) * np.ones((3, 3), dtype=np.float64)

    mascaras = [w1, w2, w3, w4, w5, w6, w7, w8, w9]
    proyecciones = [convolucion_2d_reflect(gris_u8, m) for m in mascaras]

    energia_orilla = sum(p**2 for p in proyecciones[:4])
    energia_linea = sum(p**2 for p in proyecciones[4:8])
    energia_total = sum(p**2 for p in proyecciones) + 1e-10

    # Proyección relativa (amplitud angular en el subespacio de bordes)
    medida_orilla = np.sqrt(energia_orilla / energia_total)
    medida_linea = np.sqrt(energia_linea / energia_total)

    return medida_orilla, medida_linea, proyecciones


# =============================================================================
# 5. TÉCNICA ITERATIVA DE RELAJACIÓN (RELAXATION LABELING)
# =============================================================================

def relajacion_bordes_iterativa(mag_bordes, iteraciones=3):
    """
    Refuerza iterativamente los píxeles de borde que tienen vecinos continuos
    y atenúa los píxeles aislados producto del ruido mediante soporte contextual.
    """
    p_max = float(np.max(mag_bordes))
    if p_max < 1e-6:
        return np.zeros_like(mag_bordes, dtype=np.uint8)

    # Probabilidad inicial normalizada [0.0, 1.0]
    p = np.clip(mag_bordes / p_max, 0.0, 1.0)

    # Kernel de soporte contextual en vecindad-8
    k_vec = np.array([
        [0.10, 0.15, 0.10],
        [0.15, 0.00, 0.15],
        [0.10, 0.15, 0.10]
    ], dtype=np.float64)

    for _ in range(iteraciones):
        soporte = convolucion_2d_reflect(p, k_vec)
        # Factor adaptativo: si el soporte supera 0.22 se incrementa, de lo contrario decae
        factor = 2.0 * (soporte - 0.22)
        p = np.clip(p + 0.35 * factor * p * (1.0 - p), 0.0, 1.0)

    return np.clip(np.round(p * 255.0), 0, 255).astype(np.uint8)