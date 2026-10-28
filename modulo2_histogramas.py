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


def calcular_histograma_frecuencias(canal_imagen_u8):
    """
    Calcula el conteo de frecuencias absolutas (histograma discreto)
    para cada uno de los 256 niveles de gris posibles [0..255].
    """
    conteo_histograma = [0] * 256
    alto_imagen, ancho_imagen = canal_imagen_u8.shape
    for fila in range(alto_imagen):
        for columna in range(ancho_imagen):
            intensidad = canal_imagen_u8[fila, columna]
            conteo_histograma[intensidad] += 1
    return conteo_histograma


def calcular_pdf_probabilidades(conteo_histograma, total_pixeles):
    """
    Calcula la Función de Densidad de Probabilidad (PDF) dividiendo
    cada frecuencia entre el total de píxeles (la suma total es 1.0).
    """
    if total_pixeles == 0:
        return [0.0] * 256
    return [frecuencia / float(total_pixeles) for frecuencia in conteo_histograma]


def ajustar_contraste_lineal_centrado(canal_imagen_u8, factor_contraste):
    """
    Aplica escalamiento dinámico lineal centrado en el punto medio (128):
    g(x, y) = factor_contraste * (f(x, y) - 128) + 128.
    """
    canal_escalado = factor_contraste * (canal_imagen_u8.astype(np.float64) - 128.0) + 128.0
    return np.clip(np.round(canal_escalado), 0, 255).astype(np.uint8)


def aplanar_histograma_por_vecindad(canal_imagen_u8, porcentaje_redistribucion):
    """
    Redistribución y aplanado suave del histograma sin generar bandas de falso contorno:
    1. Agrupa los píxeles por su intensidad original y desempata ordenándolos según la suma
       de sus 4 vecinos espaciales inmediatos (arriba, abajo, izquierda, derecha).
    2. Calcula la cuota ideal plana = total_pixeles / 256.
    3. Establece el techo de corte: techo = pico_maximo - factor * (pico_maximo - cuota_ideal).
    4. Poda los picos que superan el techo y acumula el exceso de píxeles.
    5. Redistribuye iterativamente el exceso entre los niveles de intensidad con déficit.
    6. Asigna cuotas enteras y mapea los píxeles al nuevo nivel de intensidad.
    """
    if porcentaje_redistribucion <= 0.0:
        return canal_imagen_u8.copy()

    filas, columnas = canal_imagen_u8.shape
    total_pixeles = filas * columnas
    factor_escala = min(1.0, max(0.0, float(porcentaje_redistribucion) / 100.0))

    # Matriz perimetral con padding reflejado para sumar los 4 vecinos ortogonales
    canal_con_padding = np.pad(canal_imagen_u8.astype(np.int32), pad_width=1, mode='edge')
    suma_vecinos_cruz = (
        canal_con_padding[:-2, 1:-1] +  # Vecino superior
        canal_con_padding[2:, 1:-1] +   # Vecino inferior
        canal_con_padding[1:-1, :-2] +  # Vecino izquierdo
        canal_con_padding[1:-1, 2:]     # Vecino derecho
    )

    # 1. Agrupamiento ordenado por intensidad y brillo local de los vecinos
    pixeles_por_intensidad = [[] for _ in range(256)]
    pixeles_planos = canal_imagen_u8.ravel()
    vecinos_planos = suma_vecinos_cruz.ravel()

    for indice in range(total_pixeles):
        intensidad_actual = pixeles_planos[indice]
        pixeles_por_intensidad[intensidad_actual].append((vecinos_planos[indice], indice))

    # Ordenar desempates locales por luminosidad de vecindad
    for nivel in range(256):
        if len(pixeles_por_intensidad[nivel]) > 1:
            pixeles_por_intensidad[nivel].sort(key=lambda elemento: elemento[0])

    # 2. Histograma inicial y cuota ideal perfectamente uniforme
    histograma_inicial = [len(pixeles_por_intensidad[i]) for i in range(256)]
    pico_maximo = max(histograma_inicial)
    cuota_ideal_uniforme = total_pixeles / 256.0

    # 3. Techo máximo permitido según el porcentaje seleccionado
    techo_maximo = pico_maximo - factor_escala * (pico_maximo - cuota_ideal_uniforme)

    # 4. Recorte de picos que exceden el techo
    histograma_redistribuido = [0.0] * 256
    exceso_pixeles_acumulado = 0.0

    for i in range(256):
        if histograma_inicial[i] > techo_maximo:
            exceso_pixeles_acumulado += (histograma_inicial[i] - techo_maximo)
            histograma_redistribuido[i] = techo_maximo
        else:
            histograma_redistribuido[i] = float(histograma_inicial[i])

    # 5. Redistribución iterativa del exceso en niveles con déficit (valles)
    while exceso_pixeles_acumulado > 0.001:
        niveles_con_deficit = [i for i in range(256) if histograma_redistribuido[i] < techo_maximo - 0.001]
        if not niveles_con_deficit:
            break
        cuota_por_nivel = exceso_pixeles_acumulado / len(niveles_con_deficit)
        exceso_pixeles_acumulado = 0.0
        for i in niveles_con_deficit:
            histograma_redistribuido[i] += cuota_por_nivel
            if histograma_redistribuido[i] > techo_maximo:
                exceso_pixeles_acumulado += (histograma_redistribuido[i] - techo_maximo)
                histograma_redistribuido[i] = techo_maximo

    # 6. Cuotas discretas enteras exactas
    cuotas_finales_enteras = [int(histograma_redistribuido[i]) for i in range(256)]
    pixeles_faltantes = total_pixeles - sum(cuotas_finales_enteras)
    for i in range(pixeles_faltantes):
        cuotas_finales_enteras[i % 256] += 1

    # 7. Asignación secuencial respetando las cuotas exactas
    imagen_redistribuida_plana = np.empty(total_pixeles, dtype=np.uint8)
    nivel_intensidad_destino = 0
    contador_en_nivel = 0

    for nivel_orig in range(256):
        for _, indice_pixel in pixeles_por_intensidad[nivel_orig]:
            while nivel_intensidad_destino < 255 and contador_en_nivel >= cuotas_finales_enteras[nivel_intensidad_destino]:
                nivel_intensidad_destino += 1
                contador_en_nivel = 0
            imagen_redistribuida_plana[indice_pixel] = nivel_intensidad_destino
            contador_en_nivel += 1

    return imagen_redistribuida_plana.reshape((filas, columnas))


# =============================================================================
# ALIAS DE COMPATIBILIDAD (Para no romper código existente)
# =============================================================================
calcular_histograma_1d = calcular_histograma_frecuencias
calcular_pdf_normalizada = calcular_pdf_probabilidades
aplicar_contraste_lineal = ajustar_contraste_lineal_centrado
redistribuir_canal_numpy = aplanar_histograma_por_vecindad
