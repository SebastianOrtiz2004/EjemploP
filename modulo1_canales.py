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


def acotar_a_uint8(valor_pixel):
    """
    Acota estrictamente cualquier valor numérico al rango entero sin signo [0, 255]
    previniendo artefactos por desbordamiento (overflow/underflow).
    """
    valor_entero = int(round(valor_pixel))
    return max(0, min(255, valor_entero))


def extraer_canales_rgb_y_gris(imagen_rgb):
    """
    Separa los tres canales espectrales (Rojo, Verde, Azul) mediante rebanado matricial
    y calcula la matriz de escala de grises fisiológica NTSC mediante ponderación ITU-R BT.601.
    
    Retorna:
        canal_rojo, canal_verde, canal_azul, imagen_escala_grises: Cuatro matrices 2D de tipo uint8.
    """
    canal_rojo = imagen_rgb[:, :, 0]
    canal_verde = imagen_rgb[:, :, 1]
    canal_azul = imagen_rgb[:, :, 2]
    
    # Ponderación fisiológica NTSC (ITU-R BT.601) según sensibilidad del ojo humano
    imagen_escala_grises = np.clip(
        np.round(0.2989 * canal_rojo + 0.5870 * canal_verde + 0.1140 * canal_azul),
        0,
        255
    ).astype(np.uint8)
    
    return canal_rojo, canal_verde, canal_azul, imagen_escala_grises


def convertir_rgb_a_flotante_normalizado(imagen_rgb):
    """
    Convierte canales uint8 [0, 255] a precisión continua double (float64)
    normalizada en el rango decimal continuo [0.0, 1.0].
    """
    canal_rojo_normalizado = imagen_rgb[:, :, 0].astype(np.float64) / 255.0
    canal_verde_normalizado = imagen_rgb[:, :, 1].astype(np.float64) / 255.0
    canal_azul_normalizado = imagen_rgb[:, :, 2].astype(np.float64) / 255.0
    return canal_rojo_normalizado, canal_verde_normalizado, canal_azul_normalizado


# =============================================================================
# ALIAS DE COMPATIBILIDAD (Para no romper código existente)
# =============================================================================
formato_uint8 = acotar_a_uint8
separar_canales_rgb = extraer_canales_rgb_y_gris
rgb_a_double = convertir_rgb_a_flotante_normalizado
