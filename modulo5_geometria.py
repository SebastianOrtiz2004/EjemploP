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


def umbral_otsu_numpy(canal_u8):
    """
    Calcula el umbral óptimo de binarización global mediante el algoritmo de Otsu
    utilizando NumPy puro (maximizando la varianza inter-clase sigma_b^2).
    """
    hist = np.bincount(canal_u8.ravel(), minlength=256).astype(np.float64)
    total_pixeles = canal_u8.size
    prob = hist / total_pixeles

    # Pesos acumulados (w0, w1)
    peso1 = np.cumsum(prob)
    peso2 = 1.0 - peso1

    # Medias acumuladas (mu0, mu1)
    intensidades = np.arange(256, dtype=np.float64)
    media_acum = np.cumsum(prob * intensidades)
    media_global = media_acum[-1]

    # Evitar división por cero
    valido = (peso1 > 0) & (peso2 > 0)
    varianza_entre_clases = np.zeros(256, dtype=np.float64)

    numerador = (media_global * peso1[valido] - media_acum[valido]) ** 2
    denominador = peso1[valido] * peso2[valido]
    varianza_entre_clases[valido] = numerador / denominador

    return int(np.argmax(varianza_entre_clases))


def imfill_holes(mascara_binaria):
    """Emula imfill(BW, 'holes') de MATLAB mediante inundación exterior con cv2.floodFill."""
    alto, ancho = mascara_binaria.shape
    inundacion = mascara_binaria.copy()
    mascara_aux = np.zeros((alto + 2, ancho + 2), dtype=np.uint8)
    cv2.floodFill(inundacion, mascara_aux, (0, 0), 255)
    inundacion_inv = cv2.bitwise_not(inundacion)
    return cv2.bitwise_or(mascara_binaria, inundacion_inv)


def calcular_propiedades_opencv(mascara_solida, etiqueta="Objeto"):
    """
    Extrae contornos perimetrales, momentos espaciales (m00, m10, m01), área, perímetro,
    Bounding Box rectangular y el código de cadena de Freeman en 8 direcciones.
    """
    contornos, _ = cv2.findContours(mascara_solida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contornos_validos = [c for c in contornos if cv2.contourArea(c) > 50]

    if not contornos_validos:
        return None

    cnt = max(contornos_validos, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    perimetro = float(cv2.arcLength(cnt, closed=True))
    x, y, w, h = cv2.boundingRect(cnt)

    m = cv2.moments(cnt)
    if m["m00"] != 0:
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
    else:
        cx, cy = x + w / 2.0, y + h / 2.0

    subimagen = mascara_solida[y:y + h, x:x + w]

    # Código de Freeman en 8-vecindad
    puntos = cnt.reshape(-1, 2)
    mapa_8 = {
        (1, 0): 0,   (1, -1): 1,  (0, -1): 2,  (-1, -1): 3,
        (-1, 0): 4,  (-1, 1): 5,  (0, 1): 6,   (1, 1): 7
    }
    freeman = []
    total = len(puntos)
    for i in range(total):
        p1, p2 = puntos[i], puntos[(i + 1) % total]
        dx = int(np.sign(p2[0] - p1[0]))
        dy = int(np.sign(p2[1] - p1[1]))
        if (dx, dy) in mapa_8:
            freeman.append(mapa_8[(dx, dy)])

    return {
        "etiqueta": etiqueta,
        "area": area,
        "perimetro": perimetro,
        "centroide": (cx, cy),
        "bbox": (x, y, w, h),
        "subimagen": subimagen,
        "contorno": cnt,
        "freeman": freeman
    }


def calcular_cuadrado_simetrico(bbox, centroide, limites):
    """
    Calcula el cuadrado envolvente simétrico: lado = max(w, h).
    Recentra sobre el centroide analítico y acota dentro de las dimensiones de la imagen.
    """
    x, y, w, h = bbox
    cx, cy = centroide
    alto_max, ancho_max = limites
    lado = max(w, h)
    xq = int(round(cx - lado / 2.0))
    yq = int(round(cy - lado / 2.0))
    xq = max(0, min(xq, ancho_max - lado))
    yq = max(0, min(yq, alto_max - lado))
    return (xq, yq, lado, lado)
