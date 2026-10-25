"""
=============================================================================
MÓDULO 2: HISTOGRAMAS, CONTRASTE LINEAL Y REDISTRIBUCIÓN DE INTENSIDAD (SIN OPENCV)
=============================================================================
Fundamento Teórico:
- Histograma Discreto: Función h(rk) = nk que cuenta la cantidad de píxeles con intensidad rk.
- PDF (Probability Density Function): p(rk) = nk / N, donde la suma total es exactamente 1.0.
- Contraste Lineal: Escalamiento dinámico centrado en 128:
  g(x,y) = factor * (f(x,y) - 128) + 128
- Redistribución Suave de Intensidad (Aplanado de Histograma):
  * Ordenamiento de vecindad: Desempate por suma de 4 vecinos (arriba, abajo, izq, der)
    para evitar grano o franjas abruptas.
  * Cuota ideal plana: total_pixeles / 256.
  * Techo de corte: techo = max_pico - factor * (max_pico - cuota_ideal).
  * Recorte de picos que exceden el techo y redistribución del exceso en valles con déficit.
=============================================================================
"""

import numpy as np


def calcular_histograma_1d(canal_u8):
    """Calcula el conteo de frecuencias absolutas para los 256 niveles de gris [0..255]."""
    hist = [0] * 256
    alto, ancho = canal_u8.shape
    for f in range(alto):
        for c in range(ancho):
            hist[canal_u8[f, c]] += 1
    return hist


def calcular_pdf_normalizada(histograma_absoluto, total_pixeles):
    """Calcula la función de densidad de probabilidad dividiendo por el total de píxeles (suma = 1.0)."""
    if total_pixeles == 0:
        return [0.0] * 256
    return [conteo / float(total_pixeles) for conteo in histograma_absoluto]


def aplicar_contraste_lineal(matriz_u8, factor_contraste):
    """Escalamiento dinámico centrado en 128: g(x,y) = factor * (f(x,y) - 128) + 128."""
    ajuste = factor_contraste * (matriz_u8.astype(np.float64) - 128.0) + 128.0
    return np.clip(np.round(ajuste), 0, 255).astype(np.uint8)


def redistribuir_canal_numpy(canal_u8, porcentaje):
    """
    Redistribución de niveles de intensidad según el algoritmo de aplanado de histograma:
    1. Agrupa coordenadas por nivel original y ordena suavemente con la suma de 4 vecinos inmediatos.
    2. Calcula el histograma original y la cuota ideal plana = total_pixeles / 256.
    3. Determina el techo de corte: techo = max_pico - factor * (max_pico - cuota_ideal).
    4. Recorta picos que superan el techo y acumula el exceso total.
    5. Redistribuye iterativamente el exceso en los niveles deficitarios (valles).
    6. Asigna cuotas enteras y mapea los píxeles ordenados por brillo local a los niveles destino.
    """
    if porcentaje <= 0.0:
        return canal_u8.copy()

    filas, columnas = canal_u8.shape
    total_pixeles = filas * columnas
    factor = min(1.0, max(0.0, float(porcentaje) / 100.0))

    # Matriz de vecinos (arriba, abajo, izquierda, derecha) usando padding reflejado
    pad = np.pad(canal_u8.astype(np.int32), pad_width=1, mode='edge')
    vecinos = (pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:])

    # 1. Agrupamiento ordenado por nivel original y suma de vecinos
    grupos = [[] for _ in range(256)]
    flat_orig = canal_u8.ravel()
    flat_vec = vecinos.ravel()

    for idx in range(total_pixeles):
        grupos[flat_orig[idx]].append((flat_vec[idx], idx))

    for nivel in range(256):
        if len(grupos[nivel]) > 1:
            grupos[nivel].sort(key=lambda item: item[0])

    # 2. Histograma original y cuota plana ideal
    hist_original = [len(grupos[i]) for i in range(256)]
    maximo_pico = max(hist_original)
    cuota_ideal = total_pixeles / 256.0

    # 3. Techo de corte según porcentaje
    techo = maximo_pico - factor * (maximo_pico - cuota_ideal)

    # 4. Recorte de picos que superan el techo
    hist_objetivo = [0.0] * 256
    exceso_total = 0.0

    for i in range(256):
        if hist_original[i] > techo:
            exceso_total += (hist_original[i] - techo)
            hist_objetivo[i] = techo
        else:
            hist_objetivo[i] = float(hist_original[i])

    # 5. Redistribución del exceso en valles
    while exceso_total > 0.001:
        niveles_deficit = [i for i in range(256) if hist_objetivo[i] < techo - 0.001]
        if not niveles_deficit:
            break
        reparto = exceso_total / len(niveles_deficit)
        exceso_total = 0.0
        for i in niveles_deficit:
            hist_objetivo[i] += reparto
            if hist_objetivo[i] > techo:
                exceso_total += (hist_objetivo[i] - techo)
                hist_objetivo[i] = techo

    # 6. Cuotas enteras exactas
    cuotas_enteras = [int(hist_objetivo[i]) for i in range(256)]
    faltante = total_pixeles - sum(cuotas_enteras)
    for i in range(faltante):
        cuotas_enteras[i % 256] += 1

    # 7. Asignación ordenada respetando las cuotas exactas
    nueva_flat = np.empty(total_pixeles, dtype=np.uint8)
    nivel_destino = 0
    puesto_en_nivel = 0

    for nivel_orig in range(256):
        for _, idx in grupos[nivel_orig]:
            while nivel_destino < 255 and puesto_en_nivel >= cuotas_enteras[nivel_destino]:
                nivel_destino += 1
                puesto_en_nivel = 0
            nueva_flat[idx] = nivel_destino
            puesto_en_nivel += 1

    return nueva_flat.reshape((filas, columnas))
