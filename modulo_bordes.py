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


def convolucion_2d_borde_reflejado(matriz_imagen, mascara_convolucion):
    """Convolución espacial 2D manual con padding reflejado en bordes (NumPy puro)."""
    alto_kernel, ancho_kernel = mascara_convolucion.shape
    relleno_alto, relleno_ancho = alto_kernel // 2, ancho_kernel // 2
    imagen_con_relleno = np.pad(
        matriz_imagen.astype(np.float64),
        ((relleno_alto, relleno_alto), (relleno_ancho, relleno_ancho)),
        mode='edge'
    )
    alto_imagen, ancho_imagen = matriz_imagen.shape
    matriz_salida = np.zeros((alto_imagen, ancho_imagen), dtype=np.float64)
    for fila_k in range(alto_kernel):
        for columna_k in range(ancho_kernel):
            matriz_salida += (
                mascara_convolucion[fila_k, columna_k]
                * imagen_con_relleno[fila_k:fila_k + alto_imagen, columna_k:columna_k + ancho_imagen]
            )
    return matriz_salida


# =============================================================================
# 1. OPERADORES DE GRADIENTE DE PRIMER ORDEN
# =============================================================================

def detectar_bordes_roberts(imagen_gris_u8):
    """Operador de Roberts: diferencias cruzadas diagonales en ventana 2x2."""
    alto_imagen, ancho_imagen = imagen_gris_u8.shape
    imagen_con_relleno = np.pad(imagen_gris_u8.astype(np.float64), ((0, 1), ((0, 1))), mode='edge')
    gradiente_diagonal_1 = (
        imagen_con_relleno[:alto_imagen, :ancho_imagen]
        - imagen_con_relleno[1:alto_imagen + 1, 1:ancho_imagen + 1]
    )
    gradiente_diagonal_2 = (
        imagen_con_relleno[:alto_imagen, 1:ancho_imagen + 1]
        - imagen_con_relleno[1:alto_imagen + 1, :ancho_imagen]
    )
    magnitud_gradiente = np.abs(gradiente_diagonal_1) + np.abs(gradiente_diagonal_2)
    return gradiente_diagonal_1, gradiente_diagonal_2, magnitud_gradiente


def detectar_bordes_prewitt(imagen_gris_u8):
    """Operador de Prewitt: derivadas ortogonales 3x3 con promediado simple."""
    mascara_prewitt_x = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
    mascara_prewitt_y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    gradiente_horizontal_x = convolucion_2d_borde_reflejado(imagen_gris_u8, mascara_prewitt_x)
    gradiente_vertical_y = convolucion_2d_borde_reflejado(imagen_gris_u8, mascara_prewitt_y)
    magnitud_gradiente = np.abs(gradiente_horizontal_x) + np.abs(gradiente_vertical_y)
    angulo_orientacion = np.degrees(np.arctan2(gradiente_vertical_y, gradiente_horizontal_x + 1e-10)) % 180
    return gradiente_horizontal_x, gradiente_vertical_y, magnitud_gradiente, angulo_orientacion


def detectar_bordes_sobel(imagen_gris_u8):
    """Operador de Sobel: derivación combinada con suavizado Gaussiano [1, 2, 1]."""
    mascara_sobel_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    mascara_sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    gradiente_horizontal_x = convolucion_2d_borde_reflejado(imagen_gris_u8, mascara_sobel_x)
    gradiente_vertical_y = convolucion_2d_borde_reflejado(imagen_gris_u8, mascara_sobel_y)
    magnitud_gradiente = np.sqrt(gradiente_horizontal_x**2 + gradiente_vertical_y**2)
    angulo_orientacion = np.degrees(np.arctan2(gradiente_vertical_y, gradiente_horizontal_x + 1e-10)) % 180
    return gradiente_horizontal_x, gradiente_vertical_y, magnitud_gradiente, angulo_orientacion


# =============================================================================
# 2. OPERADORES DIRECCIONALES DE KIRSCH (4 ORIENTACIONES)
# =============================================================================

def detectar_bordes_kirsch_4direcciones(imagen_gris_u8):
    """
    Operadores direccionales de Kirsch a 0°, 45°, 90° y 135°.
    Calcula la máxima respuesta direccional en cada píxel.
    """
    mascara_kirsch_0 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
    mascara_kirsch_45 = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]], dtype=np.float64)
    mascara_kirsch_90 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    mascara_kirsch_135 = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]], dtype=np.float64)

    respuesta_0deg = convolucion_2d_borde_reflejado(imagen_gris_u8, mascara_kirsch_0)
    respuesta_45deg = convolucion_2d_borde_reflejado(imagen_gris_u8, mascara_kirsch_45)
    respuesta_90deg = convolucion_2d_borde_reflejado(imagen_gris_u8, mascara_kirsch_90)
    respuesta_135deg = convolucion_2d_borde_reflejado(imagen_gris_u8, mascara_kirsch_135)

    pila_respuestas = np.stack(
        [np.abs(respuesta_0deg), np.abs(respuesta_45deg), np.abs(respuesta_90deg), np.abs(respuesta_135deg)],
        axis=0
    )
    magnitud_maxima_kirsch = np.max(pila_respuestas, axis=0)
    indice_direccion_dominante = np.argmax(pila_respuestas, axis=0)
    angulos_dominantes = np.array([0, 45, 90, 135])[indice_direccion_dominante]

    return (
        respuesta_0deg,
        respuesta_45deg,
        respuesta_90deg,
        respuesta_135deg,
        magnitud_maxima_kirsch,
        angulos_dominantes
    )


# =============================================================================
# 3. LAPLACIANO DE UNA GAUSSIANA (LOG / DOG) Y CRUCE POR CERO
# =============================================================================

def detectar_bordes_log_dog_cruces_cero(
    imagen_gris_u8,
    desviacion_sigma1=0.5,
    desviacion_sigma2=0.8,
    umbral_cruce_cero=3.0
):
    """
    Aproximación DoG (Diferencia de Gaussianas) al Laplaciano de una Gaussiana.
    Extrae los bordes detectando los Cruces por Cero (Zero-Crossings) con cambio de signo.
    """
    def crear_kernel_gaussiano(desviacion):
        orden_k = int(2 * math.ceil(3 * desviacion) + 1)
        if orden_k % 2 == 0:
            orden_k += 1
        centro = orden_k // 2
        y, x = np.ogrid[-centro:orden_k - centro, -centro:orden_k - centro]
        campana = np.exp(-(x**2 + y**2) / (2.0 * desviacion * desviacion))
        return campana / np.sum(campana)

    gaussiana_1 = crear_kernel_gaussiano(desviacion_sigma1)
    gaussiana_2 = crear_kernel_gaussiano(desviacion_sigma2)
    tamano_maximo = max(gaussiana_1.shape[0], gaussiana_2.shape[0])

    if gaussiana_1.shape[0] < tamano_maximo:
        tamano_relleno = (tamano_maximo - gaussiana_1.shape[0]) // 2
        gaussiana_1 = np.pad(gaussiana_1, tamano_relleno, mode='constant')
    elif gaussiana_2.shape[0] < tamano_maximo:
        tamano_relleno = (tamano_maximo - gaussiana_2.shape[0]) // 2
        gaussiana_2 = np.pad(gaussiana_2, tamano_relleno, mode='constant')

    kernel_diferencia_gaussianas = gaussiana_1 - gaussiana_2
    mapa_dog = convolucion_2d_borde_reflejado(imagen_gris_u8, kernel_diferencia_gaussianas)

    # Detección de Cruces por Cero (Zero-Crossing) en 4 direcciones espaciales
    alto_imagen, ancho_imagen = mapa_dog.shape
    mapa_cruces_por_cero = np.zeros((alto_imagen, ancho_imagen), dtype=np.uint8)

    for fila in range(1, alto_imagen - 1):
        for columna in range(1, ancho_imagen - 1):
            if (
                (mapa_dog[fila - 1, columna] * mapa_dog[fila + 1, columna] < 0 and abs(mapa_dog[fila - 1, columna] - mapa_dog[fila + 1, columna]) > umbral_cruce_cero) or
                (mapa_dog[fila, columna - 1] * mapa_dog[fila, columna + 1] < 0 and abs(mapa_dog[fila, columna - 1] - mapa_dog[fila, columna + 1]) > umbral_cruce_cero) or
                (mapa_dog[fila - 1, columna - 1] * mapa_dog[fila + 1, columna + 1] < 0 and abs(mapa_dog[fila - 1, columna - 1] - mapa_dog[fila + 1, columna + 1]) > umbral_cruce_cero) or
                (mapa_dog[fila - 1, columna + 1] * mapa_dog[fila + 1, columna - 1] < 0 and abs(mapa_dog[fila - 1, columna + 1] - mapa_dog[fila + 1, columna - 1]) > umbral_cruce_cero)
            ):
                mapa_cruces_por_cero[fila, columna] = 255

    return mapa_dog, mapa_cruces_por_cero


# =============================================================================
# 4. MÁSCARAS ORTOGONALES DE FREI-CHEN (9 MÁSCARAS)
# =============================================================================

def analizar_subespacios_frei_chen(imagen_gris_u8):
    """
    Base vectorial ortogonal de 9 máscaras de Frei-Chen:
    - 4 de Orillas (W1 a W4)
    - 4 de Líneas (W5 a W8)
    - 1 Promedio Uniforme (W9)
    Retorna la proyección de energía en el subespacio de orillas y de líneas.
    """
    raiz_de_dos = math.sqrt(2.0)
    # 4 Máscaras de Orillas (Escalones)
    mascara_orilla_1 = (1.0 / (2.0 * raiz_de_dos)) * np.array([[1, raiz_de_dos, 1], [0, 0, 0], [-1, -raiz_de_dos, -1]])
    mascara_orilla_2 = (1.0 / (2.0 * raiz_de_dos)) * np.array([[1, 0, -1], [raiz_de_dos, 0, -raiz_de_dos], [1, 0, -1]])
    mascara_orilla_3 = (1.0 / (2.0 * raiz_de_dos)) * np.array([[0, -1, raiz_de_dos], [1, 0, -1], [-raiz_de_dos, 1, 0]])
    mascara_orilla_4 = (1.0 / (2.0 * raiz_de_dos)) * np.array([[raiz_de_dos, -1, 0], [-1, 0, 1], [0, 1, -raiz_de_dos]])
    # 4 Máscaras de Líneas (Grosor 1 px)
    mascara_linea_1 = 0.5 * np.array([[0, 1, 0], [-1, 0, -1], [0, 1, 0]])
    mascara_linea_2 = 0.5 * np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]])
    mascara_linea_3 = (1.0 / 6.0) * np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
    mascara_linea_4 = (1.0 / 6.0) * np.array([[-2, 1, -2], [1, 4, 1], [-2, 1, -2]])
    # 1 Máscara Uniforme
    mascara_promedio_uniforme = (1.0 / 3.0) * np.ones((3, 3), dtype=np.float64)

    lista_mascaras = [
        mascara_orilla_1, mascara_orilla_2, mascara_orilla_3, mascara_orilla_4,
        mascara_linea_1, mascara_linea_2, mascara_linea_3, mascara_linea_4,
        mascara_promedio_uniforme
    ]
    proyecciones_subespacios = [convolucion_2d_borde_reflejado(imagen_gris_u8, m) for m in lista_mascaras]

    energia_subespacio_orillas = sum(p**2 for p in proyecciones_subespacios[:4])
    energia_subespacio_lineas = sum(p**2 for p in proyecciones_subespacios[4:8])
    energia_total_subespacio = sum(p**2 for p in proyecciones_subespacios) + 1e-10

    # Proyección relativa (energía angular en cada subespacio de Frei-Chen)
    magnitud_relativa_orillas = np.sqrt(energia_subespacio_orillas / energia_total_subespacio)
    magnitud_relativa_lineas = np.sqrt(energia_subespacio_lineas / energia_total_subespacio)

    return magnitud_relativa_orillas, magnitud_relativa_lineas, proyecciones_subespacios


# =============================================================================
# 5. TÉCNICA ITERATIVA DE RELAJACIÓN (RELAXATION LABELING)
# =============================================================================

def aplicar_relajacion_contextual_bordes(magnitud_bordes, numero_iteraciones=3):
    """
    Refuerza iterativamente los píxeles de borde que tienen vecinos continuos
    y atenúa los píxeles aislados producto del ruido mediante soporte contextual.
    """
    probabilidad_maxima = float(np.max(magnitud_bordes))
    if probabilidad_maxima < 1e-6:
        return np.zeros_like(magnitud_bordes, dtype=np.uint8)

    # Probabilidad inicial normalizada [0.0, 1.0]
    mapa_probabilidad_borde = np.clip(magnitud_bordes / probabilidad_maxima, 0.0, 1.0)

    # Kernel de soporte contextual en vecindad-8
    kernel_soporte_contextual = np.array([
        [0.10, 0.15, 0.10],
        [0.15, 0.00, 0.15],
        [0.10, 0.15, 0.10]
    ], dtype=np.float64)

    for _ in range(numero_iteraciones):
        soporte_contextual_vecindad = convolucion_2d_borde_reflejado(
            mapa_probabilidad_borde,
            kernel_soporte_contextual
        )
        # Factor adaptativo: si el soporte supera 0.22 se incrementa, de lo contrario decae
        factor_refuerzo_atenuacion = 2.0 * (soporte_contextual_vecindad - 0.22)
        mapa_probabilidad_borde = np.clip(
            mapa_probabilidad_borde
            + 0.35 * factor_refuerzo_atenuacion * mapa_probabilidad_borde * (1.0 - mapa_probabilidad_borde),
            0.0,
            1.0
        )

    return np.clip(np.round(mapa_probabilidad_borde * 255.0), 0, 255).astype(np.uint8)


# =============================================================================
# ALIAS DE COMPATIBILIDAD (Para no romper código existente)
# =============================================================================
convolucion_2d_reflect = convolucion_2d_borde_reflejado
operador_roberts_2x2 = detectar_bordes_roberts
operador_prewitt_3x3 = detectar_bordes_prewitt
operador_sobel_3x3 = detectar_bordes_sobel
operador_kirsch_4 = detectar_bordes_kirsch_4direcciones
operador_log_dog = detectar_bordes_log_dog_cruces_cero
operador_frei_chen_9 = analizar_subespacios_frei_chen
relajacion_bordes_iterativa = aplicar_relajacion_contextual_bordes