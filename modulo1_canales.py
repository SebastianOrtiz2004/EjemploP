"""
=============================================================================
MÓDULO 1: CANALES, CONVERSIÓN A DOUBLE, LUMINANCIA NTSC Y FORMATO (SIN OPENCV)
=============================================================================
Fundamento Teórico:
- Formato uint8: Representación de 8 bits sin signo con 2^8 = 256 niveles (0 a 255).
- Muestreo vs Cuantificación:
  * Muestreo: Discretización espacial del plano continuo (X, Y) en una matriz M x N.
  * Cuantificación: Discretización de la amplitud luminosa en niveles enteros discretos.
    Amplitud 0 = negro (ausencia de luz), Amplitud 255 = blanco puro (saturación).
- Luminancia NTSC (ITU-R BT.601):
  Y = 0.2989*R + 0.5870*G + 0.1140*B
  Pondera los canales según la sensibilidad biológica de la retina humana (mayor en verde).
- Flujo de Trabajo en Visión Artificial:
  Color (RGB) -> Luminancia (Escala de Grises) -> Umbralización (Binaria) -> Propiedades / Morfología.
=============================================================================
"""

import numpy as np


def formato_uint8(valor):
    """
    Acota estrictamente cualquier valor numérico al rango entero sin signo [0, 255]
    previniendo artefactos por desbordamiento (overflow/underflow).
    """
    val = int(round(valor))
    return max(0, min(255, val))


def separar_canales_rgb(matriz_rgb):
    """
    Separa los tres canales espectrales R, G, B mediante rebanado matricial
    y calcula la matriz de escala de grises fisiológica NTSC mediante NumPy puro.
    
    Retorna:
        r, g, b, gris: Cuatro matrices 2D de tipo uint8.
    """
    r = matriz_rgb[:, :, 0]
    g = matriz_rgb[:, :, 1]
    b = matriz_rgb[:, :, 2]
    # Ponderación NTSC (ITU-R BT.601)
    gris = np.clip(np.round(0.2989 * r + 0.5870 * g + 0.1140 * b), 0, 255).astype(np.uint8)
    return r, g, b, gris


def rgb_a_double(matriz_rgb):
    """
    Convierte canales uint8 [0, 255] a precisión continua double (float64) normalizada [0.0, 1.0].
    """
    r_db = matriz_rgb[:, :, 0].astype(np.float64) / 255.0
    g_db = matriz_rgb[:, :, 1].astype(np.float64) / 255.0
    b_db = matriz_rgb[:, :, 2].astype(np.float64) / 255.0
    return r_db, g_db, b_db
