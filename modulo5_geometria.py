"""
=============================================================================
MÓDULO 5: SEGMENTACIÓN, MORFOLOGÍA Y PROPIEDADES GEOMÉTRICAS (OPENCV EXCLUSIVO)
=============================================================================
Fundamento Teórico:
- Binarización (Umbralización):
  Colapsa la escala de grises a valores booleanos {0, 255}.
  * Otsu: Algoritmo estadístico que encuentra el umbral T óptimo maximizando
    la varianza inter-clase (sigma_b^2).
- Relleno Morfológico (imfill 'holes'):
  Inunda el fondo exterior desde (0, 0) e invierte el resultado para consolidar
  siluetas 100% sólidas sin cavidades internas accidentales.
- Propiedades Geométricas (RegionProps / Momentos):
  * Área (m00): Sumatoria total de píxeles encendidos: Área = sum(I(x,y)).
  * Momento en X (m10): Ponderación de los píxeles encendidos por su columna (c).
  * Momento en Y (m01): Ponderación de los píxeles encendidos por su fila (f).
  * Centroide: x_barra = m10 / m00 (columna media), y_barra = m01 / m00 (fila media).
  * Conectividad:
    - 4-conectados: Solo vecinos cardinales (N, S, E, O).
    - 8-conectados: Incluye diagonales (NE, NO, SE, SO).
  * Cuadrado Simétrico Envolvente: Lado = max(w, h), recentrado sobre el centroide:
    x_cuad = cx - lado/2, y_cuad = cy - lado/2.
  * Código de Cadena de Freeman (8-vecinos):
    Vector perimetral de direcciones discretas [0..7] con saltos angulares de 45°:
    0: Este (dx=1, dy=0), 1: Noreste, 2: Norte (dx=0, dy=-1), 3: Noroeste,
    4: Oeste (dx=-1, dy=0), 5: Suroeste, 6: Sur (dx=0, dy=1), 7: Sureste.
=============================================================================
"""

import numpy as np
import cv2


def calcular_umbral_otsu(canal_gris_u8):
    """
    Calcula el umbral óptimo de binarización global mediante el algoritmo de Otsu
    utilizando NumPy puro (maximizando la varianza inter-clase sigma_b^2).
    """
    histograma_frecuencias = np.bincount(canal_gris_u8.ravel(), minlength=256).astype(np.float64)
    total_pixeles = canal_gris_u8.size
    probabilidades_intensidad = histograma_frecuencias / total_pixeles

    # Pesos acumulados (w0, w1)
    peso_acumulado_clase0 = np.cumsum(probabilidades_intensidad)
    peso_acumulado_clase1 = 1.0 - peso_acumulado_clase0

    # Medias acumuladas (mu0, mu1)
    niveles_intensidad = np.arange(256, dtype=np.float64)
    media_acumulada = np.cumsum(probabilidades_intensidad * niveles_intensidad)
    media_global = media_acumulada[-1]

    # Evitar división por cero
    indices_validos = (peso_acumulado_clase0 > 0) & (peso_acumulado_clase1 > 0)
    varianza_interclase = np.zeros(256, dtype=np.float64)

    numerador = (media_global * peso_acumulado_clase0[indices_validos] - media_acumulada[indices_validos]) ** 2
    denominador = peso_acumulado_clase0[indices_validos] * peso_acumulado_clase1[indices_validos]
    varianza_interclase[indices_validos] = numerador / denominador

    return int(np.argmax(varianza_interclase))


def rellenar_huecos_morfologia(mascara_binaria):
    """Emula imfill(BW, 'holes') de MATLAB mediante inundación exterior con cv2.floodFill."""
    alto_mascara, ancho_mascara = mascara_binaria.shape
    imagen_inundada = mascara_binaria.copy()
    mascara_inundacion_auxiliar = np.zeros((alto_mascara + 2, ancho_mascara + 2), dtype=np.uint8)
    cv2.floodFill(imagen_inundada, mascara_inundacion_auxiliar, (0, 0), 255)
    imagen_inundada_invertida = cv2.bitwise_not(imagen_inundada)
    return cv2.bitwise_or(mascara_binaria, imagen_inundada_invertida)


def extraer_propiedades_geometricas(mascara_binaria_solida, etiqueta_objeto="Objeto"):
    """
    Extrae contornos perimetrales, momentos espaciales (m00, m10, m01), área, perímetro,
    Bounding Box rectangular y el código de cadena de Freeman en 8 direcciones.
    """
    contornos, _ = cv2.findContours(mascara_binaria_solida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contornos_validos = [c for c in contornos if cv2.contourArea(c) > 50]

    if not contornos_validos:
        return None

    contorno_principal = max(contornos_validos, key=cv2.contourArea)
    area = float(cv2.contourArea(contorno_principal))
    perimetro = float(cv2.arcLength(contorno_principal, closed=True))
    caja_x, caja_y, caja_ancho, caja_alto = cv2.boundingRect(contorno_principal)

    momentos = cv2.moments(contorno_principal)
    if momentos["m00"] != 0:
        centroide_x = momentos["m10"] / momentos["m00"]
        centroide_y = momentos["m01"] / momentos["m00"]
    else:
        centroide_x, centroide_y = caja_x + caja_ancho / 2.0, caja_y + caja_alto / 2.0

    subimagen_recortada = mascara_binaria_solida[caja_y:caja_y + caja_alto, caja_x:caja_x + caja_ancho]

    # Código de Cadena de Freeman en conectividad 8
    puntos_contorno = contorno_principal.reshape(-1, 2)
    tabla_direcciones_freeman_8 = {
        (1, 0): 0,   (1, -1): 1,  (0, -1): 2,  (-1, -1): 3,
        (-1, 0): 4,  (-1, 1): 5,  (0, 1): 6,   (1, 1): 7
    }
    codigo_cadena_freeman = []
    total_puntos = len(puntos_contorno)
    for indice in range(total_puntos):
        punto_actual = puntos_contorno[indice]
        punto_siguiente = puntos_contorno[(indice + 1) % total_puntos]
        desplazamiento_x = int(np.sign(punto_siguiente[0] - punto_actual[0]))
        desplazamiento_y = int(np.sign(punto_siguiente[1] - punto_actual[1]))
        if (desplazamiento_x, desplazamiento_y) in tabla_direcciones_freeman_8:
            codigo_cadena_freeman.append(tabla_direcciones_freeman_8[(desplazamiento_x, desplazamiento_y)])

    return {
        "etiqueta": etiqueta_objeto,
        "area": area,
        "perimetro": perimetro,
        "centroide": (centroide_x, centroide_y),
        "bbox": (caja_x, caja_y, caja_ancho, caja_alto),
        "subimagen": subimagen_recortada,
        "contorno": contorno_principal,
        "freeman": codigo_cadena_freeman
    }


def calcular_cuadrado_envolvente_simetrico(caja_delimitadora, centroide, dimensiones_limite):
    """
    Calcula el cuadrado envolvente simétrico: lado = max(w, h).
    Recentra sobre el centroide analítico y acota dentro de las dimensiones de la imagen.
    """
    caja_x, caja_y, caja_ancho, caja_alto = caja_delimitadora
    centroide_x, centroide_y = centroide
    alto_maximo, ancho_maximo = dimensiones_limite
    lado_cuadrado = max(caja_ancho, caja_alto)
    cuadrado_x = int(round(centroide_x - lado_cuadrado / 2.0))
    cuadrado_y = int(round(centroide_y - lado_cuadrado / 2.0))
    cuadrado_x = max(0, min(cuadrado_x, ancho_maximo - lado_cuadrado))
    cuadrado_y = max(0, min(cuadrado_y, alto_maximo - lado_cuadrado))
    return (cuadrado_x, cuadrado_y, lado_cuadrado, lado_cuadrado)


# =============================================================================
# ALIAS DE COMPATIBILIDAD (Para no romper código existente)
# =============================================================================
umbral_otsu_numpy = calcular_umbral_otsu
imfill_holes = rellenar_huecos_morfologia
calcular_propiedades_opencv = extraer_propiedades_geometricas
calcular_cuadrado_simetrico = calcular_cuadrado_envolvente_simetrico
