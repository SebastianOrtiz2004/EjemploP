"""
=============================================================================
SISTEMA INTEGRADO DE PROCESAMIENTO DIGITAL DE IMÁGENES (PDI) - EXAMEN PRÁCTICO
=============================================================================
Control estricto de librerías:
- Pestañas 1, 2 y 3: Implementación matemática pura con NumPy, PIL y Matplotlib.
  (Cero OpenCV en: carga, RGB, gris NTSC, histogramas, convolución con saltos,
   matrices gaussianas analíticas, FFT2, fftshift, pasa-bajos/altos y acentuado).
- Pestaña 4: Uso EXCLUSIVO de OpenCV (cv2) para delimitación, imfill, momentos,
  centroides, bounding box, cuadrado simétrico y código Freeman.
=============================================================================
"""

import os
import math
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Matplotlib para incrustar gráficos en Tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# OpenCV PERMITIDO ÚNICAMENTE para la Pestaña 4 (Análisis Geométrico y Delimitación)
import cv2


# =============================================================================
# FUNCIONES MATEMÁTICAS PURAS (MÓDULOS 1 AL 5 - CERO OPENCV)
# =============================================================================

def formato_uint8(valor):
    """Acota estrictamente cualquier valor numérico al rango entero [0, 255]."""
    val = int(round(valor))
    return max(0, min(255, val))

def separar_canales_rgb(matriz_rgb):
    """Separa canales R, G, B y calcula la escala de grises fisiológica NTSC."""
    r = matriz_rgb[:, :, 0]
    g = matriz_rgb[:, :, 1]
    b = matriz_rgb[:, :, 2]
    # Ponderación NTSC (ITU-R BT.601)
    gris = np.clip(np.round(0.2989 * r + 0.5870 * g + 0.1140 * b), 0, 255).astype(np.uint8)
    return r, g, b, gris

def calcular_histograma_1d(canal_u8):
    """Calcula el conteo de frecuencias absolutas para los 256 niveles de gris."""
    hist = [0] * 256
    alto, ancho = canal_u8.shape
    for f in range(alto):
        for c in range(ancho):
            hist[canal_u8[f, c]] += 1
    return hist

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

    # Para imágenes grandes, si el tamaño supera 400x400 trabajamos con vectorización NumPy
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
    Corrige las dimensiones de salida de forma exacta: si salto=1, salida = entrada.
    """
    k_f, k_c = mascara.shape
    pad_f = (k_f - 1) // 2 if usar_pad else 0
    pad_c = (k_c - 1) // 2 if usar_pad else 0

    alto, ancho = canal_u8.shape
    if usar_pad:
        mat_trabajo = agregar_padding_ceros(canal_u8.astype(np.float64), pad_f, pad_c)
        lim_f, lim_c = alto, ancho
        # Si hay padding, el recorrido visita exactamente de 0 a alto/ancho
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
    Permite elegir el modo de visualización del espectro (Logarítmico, Lineal crudo, Fase o Máscara H(u,v)).
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
        # Por defecto logarítmico
        espectro_vis = np.log1p(np.abs(F_shift))

    # 3. Multiplicación punto a punto y retorno con IFFT
    G_shift = F_shift * H
    G = np.fft.ifftshift(G_shift)
    img_rec = np.abs(np.fft.ifft2(G))
    return np.clip(np.round(img_rec), 0, 255).astype(np.uint8), espectro_vis



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

    # sigma_b^2 = (mu_T * w1 - mu(t))^2 / (w1 * (1 - w1))
    numerador = (media_global * peso1[valido] - media_acum[valido]) ** 2
    denominador = peso1[valido] * peso2[valido]
    varianza_entre_clases[valido] = numerador / denominador

    return int(np.argmax(varianza_entre_clases))


# =============================================================================
# MÓDULO 6: PROPIEDADES GEOMÉTRICAS Y DELIMITACIÓN (EXCLUSIVO CON OPENCV)
# =============================================================================

def imfill_holes(mascara_binaria):
    """Emula imfill(BW, 'holes') de MATLAB mediante inundación exterior."""
    alto, ancho = mascara_binaria.shape
    inundacion = mascara_binaria.copy()
    mascara_aux = np.zeros((alto + 2, ancho + 2), dtype=np.uint8)
    cv2.floodFill(inundacion, mascara_aux, (0, 0), 255)
    inundacion_inv = cv2.bitwise_not(inundacion)
    return cv2.bitwise_or(mascara_binaria, inundacion_inv)

def calcular_propiedades_opencv(mascara_solida, etiqueta="Objeto"):
    """Extrae contornos, momentos m00, m10, m01, área, perímetro, BBox y código Freeman."""
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
    """Cuadrado envolvente simétrico: lado = max(w, h), recentrado sobre el centroide."""
    x, y, w, h = bbox
    cx, cy = centroide
    alto_max, ancho_max = limites
    lado = max(w, h)
    xq = int(round(cx - lado / 2.0))
    yq = int(round(cy - lado / 2.0))
    xq = max(0, min(xq, ancho_max - lado))
    yq = max(0, min(yq, alto_max - lado))
    return (xq, yq, lado, lado)


# =============================================================================
# INTERFAZ GRÁFICA MULTIPESTAÑA CON TKINTER
# =============================================================================

class AplicacionPDIMaestra:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("Sistema de Procesamiento Digital de Imágenes - Examen Práctico")
        self.ventana.geometry("1420x860")
        self.ventana.minsize(1280, 720)
        self.ventana.configure(bg="#181818")

        # Imagen base
        self.img_rgb_original = None
        self.cargar_imagen_defecto()

        # Estilo ttk
        estilo = ttk.Style()
        try:
            estilo.theme_use("clam")
        except Exception:
            pass
        estilo.configure("TNotebook", background="#181818", borderwidth=0)
        estilo.configure("TNotebook.Tab", background="#2d2d30", foreground="#cccccc",
                        padding=[16, 7], font=("Arial", 10, "bold"))
        estilo.map("TNotebook.Tab", background=[("selected", "#0078d4")], foreground=[("selected", "#ffffff")])

        # Barra superior de carga
        top_bar = tk.Frame(self.ventana, bg="#242424", padx=12, pady=6)
        top_bar.pack(fill=tk.X)

        tk.Button(top_bar, text="📁 Cargar Imagen (Cualquier Formato)", command=self.dialogo_cargar_imagen,
                  bg="#0078d4", fg="white", font=("Arial", 9, "bold"), relief=tk.FLAT, padx=10).pack(side=tk.LEFT)

        self.lbl_info_archivo = tk.Label(top_bar, text="Imagen: Sintética Calibrada (320x420)",
                                         bg="#242424", fg="#4ec9b0", font=("Arial", 9))
        self.lbl_info_archivo.pack(side=tk.LEFT, padx=15)

        # Contenedor de Pestañas
        self.notebook = ttk.Notebook(self.ventana)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.tab1 = tk.Frame(self.notebook, bg="#181818")
        self.tab2 = tk.Frame(self.notebook, bg="#181818")
        self.tab3 = tk.Frame(self.notebook, bg="#181818")
        self.tab4 = tk.Frame(self.notebook, bg="#181818")

        self.notebook.add(self.tab1, text="  1. Descomposición RGB, Canales y 3 Histogramas  ")
        self.notebook.add(self.tab2, text="  2. Filtrado Espacial y Reducción (Stride)  ")
        self.notebook.add(self.tab3, text="  3. Frecuencia FFT2 y Acentuado  ")
        self.notebook.add(self.tab4, text="  4. Delimitación, imfill y Código Freeman  ")

        # Construir cada pestaña
        self.construir_tab1_canales_e_histogramas()
        self.construir_tab2_espacial()
        self.construir_tab3_frecuencia()
        self.construir_tab4_delimitacion()

    def cargar_imagen_defecto(self):
        """Genera imagen sintética calibrada con figuras geométricas de prueba."""
        alto, ancho = 320, 420
        self.img_rgb_original = np.full((alto, ancho, 3), 240, dtype=np.uint8)
        # Casa (Rojo)
        self.img_rgb_original[60:240, 50:180] = [215, 35, 35]
        # Auto (Amarillo)
        self.img_rgb_original[150:260, 170:340] = [225, 210, 25]

    def dialogo_cargar_imagen(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen de prueba",
            filetypes=[("Todos los formatos", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff")]
        )
        if ruta:
            try:
                img_pil = Image.open(ruta).convert("RGB")
                self.img_rgb_original = np.array(img_pil, dtype=np.uint8)
                h, w, _ = self.img_rgb_original.shape
                self.lbl_info_archivo.config(text=f"Imagen: {os.path.basename(ruta)} ({w}x{h} px)")
                # Actualizar automáticamente todas las pestañas
                self.actualizar_tab1()
                self.actualizar_tab2_preview()
                self.actualizar_tab3_ffts()
                self.actualizar_tab4_delimitacion()
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar la imagen:\n{e}")

    # =========================================================================
    # PESTAÑA 1: IMÁGENES RGB, CANALES EN GRIS Y 3 HISTOGRAMAS
    # =========================================================================
    def construir_tab1_canales_e_histogramas(self):
        panel_ctrl = tk.Frame(self.tab1, bg="#242424", width=250, padx=8, pady=6)
        panel_ctrl.pack(side=tk.LEFT, fill=tk.Y)
        panel_ctrl.pack_propagate(False)

        tk.Label(panel_ctrl, text="Controles por Canal (R, G, B)", bg="#242424", fg="#ffd700",
                 font=("Arial", 9, "bold")).pack(anchor="w", pady=(0, 2))

        # --- CANAL ROJO ---
        f_r = tk.LabelFrame(panel_ctrl, text="🔴 Canal Rojo", bg="#202020", fg="#ff6b6b",
                            font=("Arial", 8, "bold"), padx=4, pady=2)
        f_r.pack(fill=tk.X, pady=2)

        tk.Label(f_r, text="Contraste R:", bg="#202020", fg="white", font=("Arial", 8)).pack(anchor="w")
        self.slider_c_r = tk.Scale(f_r, from_=0.2, to=3.0, resolution=0.05, orient=tk.HORIZONTAL,
                                   bg="#202020", fg="white", highlightthickness=0,
                                   activebackground="#ff6b6b", troughcolor="#111111")
        self.slider_c_r.set(1.0)
        self.slider_c_r.pack(fill=tk.X)
        self.slider_c_r.bind("<ButtonRelease-1>", lambda _: self.actualizar_tab1())

        tk.Label(f_r, text="Redistribución R (%):", bg="#202020", fg="white", font=("Arial", 8)).pack(anchor="w")
        self.slider_red_r = tk.Scale(f_r, from_=0, to=100, resolution=5, orient=tk.HORIZONTAL,
                                     bg="#202020", fg="#ff6b6b", highlightthickness=0,
                                     activebackground="#ff6b6b", troughcolor="#111111")
        self.slider_red_r.set(0)
        self.slider_red_r.pack(fill=tk.X)
        self.slider_red_r.bind("<ButtonRelease-1>", lambda _: self.actualizar_tab1())

        # --- CANAL VERDE ---
        f_g = tk.LabelFrame(panel_ctrl, text="🟢 Canal Verde", bg="#202020", fg="#51cf66",
                            font=("Arial", 8, "bold"), padx=4, pady=2)
        f_g.pack(fill=tk.X, pady=2)

        tk.Label(f_g, text="Contraste G:", bg="#202020", fg="white", font=("Arial", 8)).pack(anchor="w")
        self.slider_c_g = tk.Scale(f_g, from_=0.2, to=3.0, resolution=0.05, orient=tk.HORIZONTAL,
                                   bg="#202020", fg="white", highlightthickness=0,
                                   activebackground="#51cf66", troughcolor="#111111")
        self.slider_c_g.set(1.0)
        self.slider_c_g.pack(fill=tk.X)
        self.slider_c_g.bind("<ButtonRelease-1>", lambda _: self.actualizar_tab1())

        tk.Label(f_g, text="Redistribución G (%):", bg="#202020", fg="white", font=("Arial", 8)).pack(anchor="w")
        self.slider_red_g = tk.Scale(f_g, from_=0, to=100, resolution=5, orient=tk.HORIZONTAL,
                                     bg="#202020", fg="#51cf66", highlightthickness=0,
                                     activebackground="#51cf66", troughcolor="#111111")
        self.slider_red_g.set(0)
        self.slider_red_g.pack(fill=tk.X)
        self.slider_red_g.bind("<ButtonRelease-1>", lambda _: self.actualizar_tab1())

        # --- CANAL AZUL ---
        f_b = tk.LabelFrame(panel_ctrl, text="🔵 Canal Azul", bg="#202020", fg="#339af0",
                            font=("Arial", 8, "bold"), padx=4, pady=2)
        f_b.pack(fill=tk.X, pady=2)

        tk.Label(f_b, text="Contraste B:", bg="#202020", fg="white", font=("Arial", 8)).pack(anchor="w")
        self.slider_c_b = tk.Scale(f_b, from_=0.2, to=3.0, resolution=0.05, orient=tk.HORIZONTAL,
                                   bg="#202020", fg="white", highlightthickness=0,
                                   activebackground="#339af0", troughcolor="#111111")
        self.slider_c_b.set(1.0)
        self.slider_c_b.pack(fill=tk.X)
        self.slider_c_b.bind("<ButtonRelease-1>", lambda _: self.actualizar_tab1())

        tk.Label(f_b, text="Redistribución B (%):", bg="#202020", fg="white", font=("Arial", 8)).pack(anchor="w")
        self.slider_red_b = tk.Scale(f_b, from_=0, to=100, resolution=5, orient=tk.HORIZONTAL,
                                     bg="#202020", fg="#339af0", highlightthickness=0,
                                     activebackground="#339af0", troughcolor="#111111")
        self.slider_red_b.set(0)
        self.slider_red_b.pack(fill=tk.X)
        self.slider_red_b.bind("<ButtonRelease-1>", lambda _: self.actualizar_tab1())

        # Botón para restaurar
        tk.Button(panel_ctrl, text="↺ Restaurar Canales", command=self.reset_tab1_valores,
                  bg="#3e3e42", fg="white", font=("Arial", 8, "bold"), relief=tk.FLAT).pack(fill=tk.X, pady=4)

        # Referencias Teóricas
        info_teoria = (
            "• Luminancia NTSC:\n"
            "  Y = 0.299R + 0.587G + 0.114B\n"
            "• Espectro:\n"
            "  R: 600-700nm | G: 500-600nm | B: 400-500nm\n"
            "• Formato: uint8 (0-255)"
        )
        tk.Label(panel_ctrl, text=info_teoria, bg="#141414", fg="#ffd700",
                 font=("Consolas", 7), justify=tk.LEFT, padx=4, pady=3).pack(fill=tk.X, pady=(2, 0))

        # Panel principal derecho
        panel_derecho = tk.Frame(self.tab1, bg="#181818", padx=4, pady=2)
        panel_derecho.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # SECCIÓN SUPERIOR: GALERÍA DE 5 IMÁGENES MEDIANTE GRID HOMOGÉNEO
        frame_galeria = tk.Frame(panel_derecho, bg="#1c1c1c", padx=3, pady=3)
        frame_galeria.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 2))

        for col_idx in range(5):
            frame_galeria.columnconfigure(col_idx, weight=1, uniform="galeria_col")
        frame_galeria.rowconfigure(0, weight=1)

        # Tarjeta 1: Original RGB
        tarjeta_orig = tk.Frame(frame_galeria, bg="#181818", padx=2, pady=2)
        tarjeta_orig.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        tk.Label(tarjeta_orig, text="1. Original (RGB)", bg="#181818", fg="white", font=("Arial", 8, "bold")).pack()
        self.lbl_v_orig = tk.Label(tarjeta_orig, bg="#0d0d0d")
        self.lbl_v_orig.pack(expand=True, fill=tk.BOTH, pady=2)

        # Tarjeta 2: Resultante RGB
        tarjeta_res = tk.Frame(frame_galeria, bg="#181818", padx=2, pady=2)
        tarjeta_res.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        tk.Label(tarjeta_res, text="2. Resultante (RGB)", bg="#181818", fg="#4ec9b0", font=("Arial", 8, "bold")).pack()
        self.lbl_v_res = tk.Label(tarjeta_res, bg="#0d0d0d")
        self.lbl_v_res.pack(expand=True, fill=tk.BOTH, pady=2)

        # Tarjeta 3: Canal Rojo (Gris)
        tarjeta_r = tk.Frame(frame_galeria, bg="#181818", padx=2, pady=2)
        tarjeta_r.grid(row=0, column=2, sticky="nsew", padx=2, pady=2)
        tk.Label(tarjeta_r, text="3. Canal Rojo (Gris)", bg="#181818", fg="#ff6b6b", font=("Arial", 8, "bold")).pack()
        self.lbl_v_r = tk.Label(tarjeta_r, bg="#0d0d0d")
        self.lbl_v_r.pack(expand=True, fill=tk.BOTH, pady=2)

        # Tarjeta 4: Canal Verde (Gris)
        tarjeta_g = tk.Frame(frame_galeria, bg="#181818", padx=2, pady=2)
        tarjeta_g.grid(row=0, column=3, sticky="nsew", padx=2, pady=2)
        tk.Label(tarjeta_g, text="4. Canal Verde (Gris)", bg="#181818", fg="#51cf66", font=("Arial", 8, "bold")).pack()
        self.lbl_v_g = tk.Label(tarjeta_g, bg="#0d0d0d")
        self.lbl_v_g.pack(expand=True, fill=tk.BOTH, pady=2)

        # Tarjeta 5: Canal Azul (Gris)
        tarjeta_b = tk.Frame(frame_galeria, bg="#181818", padx=2, pady=2)
        tarjeta_b.grid(row=0, column=4, sticky="nsew", padx=2, pady=2)
        tk.Label(tarjeta_b, text="5. Canal Azul (Gris)", bg="#181818", fg="#339af0", font=("Arial", 8, "bold")).pack()
        self.lbl_v_b = tk.Label(tarjeta_b, bg="#0d0d0d")
        self.lbl_v_b.pack(expand=True, fill=tk.BOTH, pady=2)

        # SECCIÓN INFERIOR: 3 HISTOGRAMAS SEPARADOS (R, G, B)
        frame_hist = tk.Frame(panel_derecho, bg="#181818", height=240)
        frame_hist.pack(side=tk.BOTTOM, fill=tk.X, pady=(2, 0))
        frame_hist.pack_propagate(False)

        self.fig_hist, self.axes_hist = plt.subplots(1, 3, figsize=(10, 2.2))
        self.fig_hist.patch.set_facecolor("#181818")
        self.canvas_hist = FigureCanvasTkAgg(self.fig_hist, master=frame_hist)
        self.canvas_hist.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.actualizar_tab1()

    def reset_tab1_valores(self):
        self.slider_c_r.set(1.0)
        self.slider_red_r.set(0)
        self.slider_c_g.set(1.0)
        self.slider_red_g.set(0)
        self.slider_c_b.set(1.0)
        self.slider_red_b.set(0)
        self.actualizar_tab1()

    def actualizar_tab1(self):
        c_r = float(self.slider_c_r.get())
        red_r = float(self.slider_red_r.get())
        c_g = float(self.slider_c_g.get())
        red_g = float(self.slider_red_g.get())
        c_b = float(self.slider_c_b.get())
        red_b = float(self.slider_red_b.get())

        r, g, b, _ = separar_canales_rgb(self.img_rgb_original)

        # Aplicar contraste y redistribución analítica INDEPENDIENTE por canal
        r_mod = redistribuir_canal_numpy(aplicar_contraste_lineal(r, c_r), red_r)
        g_mod = redistribuir_canal_numpy(aplicar_contraste_lineal(g, c_g), red_g)
        b_mod = redistribuir_canal_numpy(aplicar_contraste_lineal(b, c_b), red_b)
        img_resultante_rgb = np.stack([r_mod, g_mod, b_mod], axis=-1)

        # Dimensiones uniformes para las 5 tarjetas de imagen
        for matriz, modo, label in [
            (self.img_rgb_original, "RGB", self.lbl_v_orig),
            (img_resultante_rgb, "RGB", self.lbl_v_res),
            (r_mod, "L", self.lbl_v_r),
            (g_mod, "L", self.lbl_v_g),
            (b_mod, "L", self.lbl_v_b)
        ]:
            p_img = Image.fromarray(matriz, mode=modo)
            p_img.thumbnail((190, 150), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(p_img)
            label.config(image=tk_img)
            label.image = tk_img

        # Calcular histogramas originales y modificados
        h_r_orig, h_r_mod = calcular_histograma_1d(r), calcular_histograma_1d(r_mod)
        h_g_orig, h_g_mod = calcular_histograma_1d(g), calcular_histograma_1d(g_mod)
        h_b_orig, h_b_mod = calcular_histograma_1d(b), calcular_histograma_1d(b_mod)

        canales_info = [
            (self.axes_hist[0], "#ff4444", "Canal Rojo", h_r_orig, h_r_mod),
            (self.axes_hist[1], "#00cc66", "Canal Verde", h_g_orig, h_g_mod),
            (self.axes_hist[2], "#3399ff", "Canal Azul", h_b_orig, h_b_mod)
        ]

        for ax, col, nombre, h_orig, h_mod in canales_info:
            ax.clear()
            ax.set_facecolor("#111111")
            ax.tick_params(colors="gray", labelsize=7)
            ax.plot(range(256), h_orig, color="#777777", linestyle="--", linewidth=1.0, label="Original")
            ax.bar(range(256), h_mod, color=col, width=1.0, alpha=0.85, label="Modificado")
            ax.set_title(f"Histograma {nombre}", color="white", fontsize=8, weight="bold")
            ax.legend(loc="upper right", fontsize=6, facecolor="#222222", edgecolor="none", labelcolor="white")

        self.fig_hist.subplots_adjust(left=0.06, right=0.98, top=0.85, bottom=0.18, wspace=0.20)
        self.canvas_hist.draw()

    # =========================================================================
    # PESTAÑA 2: FILTRADO ESPACIAL Y REDUCCIÓN POR SALTO (STRIDE)
    # =========================================================================
    def construir_tab2_espacial(self):
        panel_ctrl = tk.Frame(self.tab2, bg="#242424", width=340, padx=10, pady=8)
        panel_ctrl.pack(side=tk.LEFT, fill=tk.Y)
        panel_ctrl.pack_propagate(False)

        # 1. Ruido
        tk.Label(panel_ctrl, text="1. Inyección de Ruido Sal y Pimienta:", bg="#242424", fg="#ffd700",
                 font=("Arial", 9, "bold")).pack(anchor="w", pady=(0, 2))
        f_ruido = tk.Frame(panel_ctrl, bg="#242424")
        f_ruido.pack(fill=tk.X, pady=(0, 4))
        self.slider_ruido = tk.Scale(f_ruido, from_=0.0, to=0.5, resolution=0.02, orient=tk.HORIZONTAL,
                                     bg="#242424", fg="white", highlightthickness=0, troughcolor="#111111")
        self.slider_ruido.set(0.05)
        self.slider_ruido.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 2. Selector de Filtro
        tk.Label(panel_ctrl, text="2. Técnica de Filtrado:", bg="#242424", fg="white",
                 font=("Arial", 9, "bold")).pack(anchor="w", pady=(6, 2))
        self.combo_filtro_espacial = ttk.Combobox(
            panel_ctrl, values=["Lineal (Gaussiana Calculada)", "Lineal (Media / Promedio)",
                                "No Lineal (Mediana)"],
            state="readonly", font=("Arial", 9)
        )
        self.combo_filtro_espacial.set("Lineal (Gaussiana Calculada)")
        self.combo_filtro_espacial.pack(fill=tk.X)
        self.combo_filtro_espacial.bind("<<ComboboxSelected>>", lambda _: self.actualizar_matriz_text())

        # 3. Tamaño impar de máscara
        tk.Label(panel_ctrl, text="3. Dimensión Impar de Máscara (KxK):", bg="#242424", fg="white",
                 font=("Arial", 9, "bold")).pack(anchor="w", pady=(6, 2))
        f_tam = tk.Frame(panel_ctrl, bg="#242424")
        f_tam.pack(fill=tk.X)
        self.entry_tam_k = tk.Entry(f_tam, width=5, justify="center", font=("Arial", 9, "bold"))
        self.entry_tam_k.insert(0, "3")
        self.entry_tam_k.pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(f_tam, text="Recalcular Núcleo", command=self.actualizar_matriz_text,
                  bg="#3e3e42", fg="white", font=("Arial", 8, "bold"), relief=tk.FLAT).pack(side=tk.LEFT)

        # 4. Inspección de la Máscara
        tk.Label(panel_ctrl, text="4. Valores del Núcleo / Máscara W(x,y):", bg="#242424", fg="#4ec9b0",
                 font=("Arial", 9, "bold")).pack(anchor="w", pady=(8, 2))
        self.text_matriz = tk.Text(panel_ctrl, height=6, bg="#111111", fg="#ffd700",
                                   font=("Consolas", 8), relief=tk.SOLID, bd=1)
        self.text_matriz.pack(fill=tk.X)

        # 5. Salto (Stride) y Padding
        tk.Label(panel_ctrl, text="5. Parámetros de Reducción:", bg="#242424", fg="white",
                 font=("Arial", 9, "bold")).pack(anchor="w", pady=(8, 2))
        f_saltos = tk.Frame(panel_ctrl, bg="#242424")
        f_saltos.pack(fill=tk.X)
        tk.Label(f_saltos, text="Salto Filas:", bg="#242424", fg="lightgray", font=("Arial", 8)).grid(row=0, column=0)
        self.entry_salto_f = tk.Entry(f_saltos, width=4, justify="center")
        self.entry_salto_f.insert(0, "1")
        self.entry_salto_f.grid(row=0, column=1, padx=4)

        tk.Label(f_saltos, text="Salto Cols:", bg="#242424", fg="lightgray", font=("Arial", 8)).grid(row=0, column=2)
        self.entry_salto_c = tk.Entry(f_saltos, width=4, justify="center")
        self.entry_salto_c.insert(0, "1")
        self.entry_salto_c.grid(row=0, column=3, padx=4)

        self.var_padding = tk.BooleanVar(value=True)
        tk.Checkbutton(panel_ctrl, text="Aplicar Padding de Ceros", variable=self.var_padding,
                       bg="#242424", fg="white", selectcolor="#111111", font=("Arial", 9)).pack(anchor="w", pady=4)

        tk.Button(panel_ctrl, text="⚡ Ejecutar Convolución y Salto", command=self.ejecutar_filtro_espacial,
                  bg="#107c41", fg="white", font=("Arial", 10, "bold"), relief=tk.FLAT).pack(fill=tk.X, pady=8)

        # Panel de visualizadores
        panel_vis = tk.Frame(self.tab2, bg="#181818", padx=10, pady=10)
        panel_vis.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        panel_vis.columnconfigure(0, weight=1)
        panel_vis.columnconfigure(1, weight=1)
        panel_vis.rowconfigure(0, weight=1)

        f_izq = tk.Frame(panel_vis, bg="#202020", padx=6, pady=6)
        f_izq.grid(row=0, column=0, sticky="nsew", padx=4)
        tk.Label(f_izq, text="Imagen con Ruido f(x,y)", bg="#202020", fg="white", font=("Arial", 10, "bold")).pack()
        self.lbl_dim_esp_orig = tk.Label(f_izq, text="", bg="#202020", fg="gray", font=("Arial", 8))
        self.lbl_dim_esp_orig.pack()
        self.lbl_img_esp_orig = tk.Label(f_izq, bg="#0d0d0d")
        self.lbl_img_esp_orig.pack(expand=True, fill=tk.BOTH, pady=4)

        f_der = tk.Frame(panel_vis, bg="#202020", padx=6, pady=6)
        f_der.grid(row=0, column=1, sticky="nsew", padx=4)
        tk.Label(f_der, text="Imagen Filtrada y Reducida g(x,y)", bg="#202020", fg="#4ec9b0", font=("Arial", 10, "bold")).pack()
        self.lbl_dim_esp_res = tk.Label(f_der, text="", bg="#202020", fg="#4ec9b0", font=("Arial", 8, "bold"))
        self.lbl_dim_esp_res.pack()
        self.lbl_img_esp_res = tk.Label(f_der, bg="#0d0d0d")
        self.lbl_img_esp_res.pack(expand=True, fill=tk.BOTH, pady=4)

        self.actualizar_matriz_text()
        self.actualizar_tab2_preview()

    def actualizar_matriz_text(self):
        try:
            k = int(self.entry_tam_k.get())
            if k < 3 or k % 2 == 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Atención", "El orden de la máscara debe ser impar (3, 5, 7...).")
            self.entry_tam_k.delete(0, tk.END)
            self.entry_tam_k.insert(0, "3")
            k = 3

        tipo = self.combo_filtro_espacial.get()
        self.text_matriz.delete("1.0", tk.END)

        if "Gaussiana" in tipo:
            w = calcular_matriz_gaussiana(k, sigma=1.0)
            texto = f"Gaussiana analítica (k={k}, suma=1.000):\n"
            for fila in w:
                texto += "[" + " ".join([f"{val:6.4f}" for val in fila]) + "]\n"
        elif "Media" in tipo:
            w = np.ones((k, k), dtype=np.float64) / float(k * k)
            texto = f"Media aritmética uniforme (k={k}, peso=1/{k*k}):\n"
            for fila in w:
                texto += "[" + " ".join([f"{val:6.4f}" for val in fila]) + "]\n"
        else:
            texto = f"Filtro No Lineal ({tipo}):\nOpera por ordenamiento local de la vecindad {k}x{k}."

        self.text_matriz.insert(tk.END, texto)

    def actualizar_tab2_preview(self):
        dens = float(self.slider_ruido.get())
        self.img_ruidosa = inyectar_ruido_sal_pimienta(self.img_rgb_original, dens)
        h, w, _ = self.img_ruidosa.shape
        self.lbl_dim_esp_orig.config(text=f"{h}x{w} px | Ruido: {dens*100:.1f}%")

        im_pil = Image.fromarray(self.img_ruidosa)
        im_pil.thumbnail((440, 440), Image.Resampling.LANCZOS)
        self.tk_ruidosa = ImageTk.PhotoImage(im_pil)
        self.lbl_img_esp_orig.config(image=self.tk_ruidosa)

    def ejecutar_filtro_espacial(self):
        self.actualizar_tab2_preview()
        try:
            k = int(self.entry_tam_k.get())
            salto_f = max(1, int(self.entry_salto_f.get()))
            salto_c = max(1, int(self.entry_salto_c.get()))
        except ValueError:
            salto_f, salto_c = 1, 1
            
        pad = self.var_padding.get()
        tipo = self.combo_filtro_espacial.get()

        r, g, b, _ = separar_canales_rgb(self.img_ruidosa)

        if "Gaussiana" in tipo:
            mascara = calcular_matriz_gaussiana(k)
            r_out = convolucion_con_salto_manual(r, mascara, salto_f, salto_c, pad)
            g_out = convolucion_con_salto_manual(g, mascara, salto_f, salto_c, pad)
            b_out = convolucion_con_salto_manual(b, mascara, salto_f, salto_c, pad)
        elif "Media" in tipo:
            mascara = np.ones((k, k), dtype=np.float64) / float(k * k)
            r_out = convolucion_con_salto_manual(r, mascara, salto_f, salto_c, pad)
            g_out = convolucion_con_salto_manual(g, mascara, salto_f, salto_c, pad)
            b_out = convolucion_con_salto_manual(b, mascara, salto_f, salto_c, pad)
        else:
            # Mediana
            r_out = filtro_mediana_manual(r, k, salto_f, salto_c)
            g_out = filtro_mediana_manual(g, k, salto_f, salto_c)
            b_out = filtro_mediana_manual(b, k, salto_f, salto_c)

        img_red = np.stack([r_out, g_out, b_out], axis=-1)
        h_r, w_r, _ = img_red.shape
        
        texto_reduccion = f"{h_r}x{w_r} px (Sin reducción)" if (salto_f == 1 and salto_c == 1) else f"{h_r}x{w_r} px (Reducida por salto {salto_f}x{salto_c})"
        self.lbl_dim_esp_res.config(text=texto_reduccion)

        im_pil = Image.fromarray(img_red)
        im_pil.thumbnail((440, 440), Image.Resampling.LANCZOS)
        self.tk_reducida = ImageTk.PhotoImage(im_pil)
        self.lbl_img_esp_res.config(image=self.tk_reducida)

    # =========================================================================
    # PESTAÑA 3: FRECUENCIA (FFT 2D CON RUIDO) Y ACENTUADO ESPACIAL
    # =========================================================================
    def construir_tab3_frecuencia(self):
        panel_ctrl = tk.Frame(self.tab3, bg="#242424", width=310, padx=10, pady=8)
        panel_ctrl.pack(side=tk.LEFT, fill=tk.Y)
        panel_ctrl.pack_propagate(False)

        tk.Label(panel_ctrl, text="Filtrado en Frecuencia (FFT)", bg="#242424", fg="#ffd700",
                 font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 4))

        # Ruido para la FFT (hasta 100%)
        tk.Label(panel_ctrl, text="1. Ruido en Imagen de Entrada (%):", bg="#242424", fg="white", font=("Arial", 9, "bold")).pack(anchor="w", pady=(4, 1))
        self.slider_ruido_fft = tk.Scale(panel_ctrl, from_=0, to=100, resolution=1, orient=tk.HORIZONTAL,
                                         bg="#242424", fg="white", highlightthickness=0, troughcolor="#111111")
        self.slider_ruido_fft.set(0)
        self.slider_ruido_fft.pack(fill=tk.X)

        # Filtro Ideal
        tk.Label(panel_ctrl, text="2. Filtro Ideal Frecuencial:", bg="#242424", fg="white", font=("Arial", 9, "bold")).pack(anchor="w", pady=(6, 1))
        self.combo_tipo_fft = ttk.Combobox(panel_ctrl, values=["pasa_bajos", "pasa_altos"], state="readonly")
        self.combo_tipo_fft.set("pasa_bajos")
        self.combo_tipo_fft.pack(fill=tk.X)

        # Radio D0
        tk.Label(panel_ctrl, text="3. Radio de Corte D0 (px):", bg="#242424", fg="white", font=("Arial", 9, "bold")).pack(anchor="w", pady=(6, 1))
        self.slider_d0 = tk.Scale(panel_ctrl, from_=5, to=150, resolution=5, orient=tk.HORIZONTAL,
                                  bg="#242424", fg="white", highlightthickness=0, troughcolor="#111111")
        self.slider_d0.set(40)
        self.slider_d0.pack(fill=tk.X)

        # Selector de Modo del Espectro
        tk.Label(panel_ctrl, text="4. Visualización del Espectro:", bg="#242424", fg="#4ec9b0", font=("Arial", 9, "bold")).pack(anchor="w", pady=(8, 1))
        self.combo_modo_espectro = ttk.Combobox(
            panel_ctrl, values=["logaritmico", "lineal", "fase", "mascara_filtro_H"], state="readonly"
        )
        self.combo_modo_espectro.set("logaritmico")
        self.combo_modo_espectro.pack(fill=tk.X)

        # Acentuado
        tk.Label(panel_ctrl, text="5. Acentuado sobre Imagen IFFT:", bg="#242424", fg="#ffd700",
                 font=("Arial", 10, "bold")).pack(anchor="w", pady=(12, 2))

        self.var_aplicar_acentuado = tk.BooleanVar(value=False)
        tk.Checkbutton(panel_ctrl, text="Aplicar Acentuado Espacial", variable=self.var_aplicar_acentuado,
                       bg="#242424", fg="white", selectcolor="#111111", font=("Arial", 9, "bold")).pack(anchor="w")

        self.combo_modo_acentuado = ttk.Combobox(panel_ctrl, values=["high_boost", "solo_bordes"], state="readonly")
        self.combo_modo_acentuado.set("high_boost")
        self.combo_modo_acentuado.pack(fill=tk.X, pady=4)

        tk.Label(panel_ctrl, text="Factor A (High-Boost):", bg="#242424", fg="lightgray", font=("Arial", 8)).pack(anchor="w")
        self.slider_factor_a = tk.Scale(panel_ctrl, from_=1.0, to=2.5, resolution=0.1, orient=tk.HORIZONTAL,
                                        bg="#242424", fg="white", highlightthickness=0, troughcolor="#111111")
        self.slider_factor_a.set(1.2)
        self.slider_factor_a.pack(fill=tk.X)

        tk.Button(panel_ctrl, text="⚡ Ejecutar FFT + Acentuado", command=self.actualizar_tab3_ffts,
                  bg="#107c41", fg="white", font=("Arial", 10, "bold"), relief=tk.FLAT).pack(fill=tk.X, pady=12)

        # 4 visualizadores
        panel_vis = tk.Frame(self.tab3, bg="#181818", padx=6, pady=6)
        panel_vis.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        panel_vis.columnconfigure(0, weight=1)
        panel_vis.columnconfigure(1, weight=1)
        panel_vis.rowconfigure(0, weight=1)
        panel_vis.rowconfigure(1, weight=1)

        f_c1 = tk.Frame(panel_vis, bg="#202020", padx=4, pady=4)
        f_c1.grid(row=0, column=0, sticky="nsew", padx=3, pady=3)
        self.lbl_tit_espectro = tk.Label(f_c1, text="Espectro / Máscara", bg="#202020", fg="#ffd700", font=("Arial", 9, "bold"))
        self.lbl_tit_espectro.pack()
        self.lbl_img_fft_espectro = tk.Label(f_c1, bg="#0d0d0d")
        self.lbl_img_fft_espectro.pack(expand=True, fill=tk.BOTH)

        f_c2 = tk.Frame(panel_vis, bg="#202020", padx=4, pady=4)
        f_c2.grid(row=0, column=1, sticky="nsew", padx=3, pady=3)
        self.lbl_tit_ifft = tk.Label(f_c2, text="Salida IFFT 2D (Color)", bg="#202020", fg="white", font=("Arial", 9, "bold"))
        self.lbl_tit_ifft.pack()
        self.lbl_img_fft_ifft = tk.Label(f_c2, bg="#0d0d0d")
        self.lbl_img_fft_ifft.pack(expand=True, fill=tk.BOTH)

        f_c3 = tk.Frame(panel_vis, bg="#202020", padx=4, pady=4)
        f_c3.grid(row=1, column=0, sticky="nsew", padx=3, pady=3)
        self.lbl_tit_entrada = tk.Label(f_c3, text="Entrada Original a Color f(x,y)", bg="#202020", fg="white", font=("Arial", 9, "bold"))
        self.lbl_tit_entrada.pack()
        self.lbl_img_fft_base = tk.Label(f_c3, bg="#0d0d0d")
        self.lbl_img_fft_base.pack(expand=True, fill=tk.BOTH)

        f_c4 = tk.Frame(panel_vis, bg="#202020", padx=4, pady=4)
        f_c4.grid(row=1, column=1, sticky="nsew", padx=3, pady=3)
        self.lbl_tit_acentuado = tk.Label(f_c4, text="Resultado Acentuado (Color)", bg="#202020", fg="#4ec9b0", font=("Arial", 9, "bold"))
        self.lbl_tit_acentuado.pack()
        self.lbl_img_fft_acentuado = tk.Label(f_c4, bg="#0d0d0d")
        self.lbl_img_fft_acentuado.pack(expand=True, fill=tk.BOTH)

        self.actualizar_tab3_ffts()

    def actualizar_tab3_ffts(self):
        tipo_fft = self.combo_tipo_fft.get()
        d0 = float(self.slider_d0.get())
        # Slider en escala 0 a 100% -> densidad 0.0 a 1.0
        val_ruido = float(self.slider_ruido_fft.get())
        dens_ruido = val_ruido / 100.0 if val_ruido > 1.0 else val_ruido
        porcentaje_ruido = dens_ruido * 100.0

        modo_esp = self.combo_modo_espectro.get()

        # Imagen de entrada original a COLOR con ruido inyectado si se solicita
        img_color_ruidosa = inyectar_ruido_sal_pimienta(self.img_rgb_original, dens_ruido)

        # 1. FFT y Filtrado Frecuencial a Color (canal R, G, B procesados)
        img_ifft_color, espectro = filtrado_frecuencia_fft(img_color_ruidosa, tipo=tipo_fft, d0=d0, modo_espectro=modo_esp)

        # Normalización de visualización del espectro a [0, 255]
        min_e, max_e = np.min(espectro), np.max(espectro)
        if max_e > min_e:
            esp_norm = (255.0 * (espectro - min_e) / (max_e - min_e)).astype(np.uint8)
        else:
            esp_norm = np.zeros_like(espectro, dtype=np.uint8)

        # 2. Acentuado a Color (si está activo)
        aplicar_ac = self.var_aplicar_acentuado.get()
        modo_ac = self.combo_modo_acentuado.get()
        factor_a = float(self.slider_factor_a.get())

        if aplicar_ac:
            img_ac_color = acentuado_espacial(img_ifft_color, modo=modo_ac, factor_a=factor_a)
            tit_ac = f"Acentuado Color: {modo_ac.upper()} (A={factor_a:.1f})"
        else:
            img_ac_color = img_ifft_color
            tit_ac = "Acentuado Inactivo"

        etiqueta_esp = "MÁSCARA FILTRO H(u,v)" if "mascara" in modo_esp else modo_esp.upper()
        self.lbl_tit_espectro.config(text=f"Espectro: {etiqueta_esp}")
        self.lbl_tit_entrada.config(text=f"Entrada a Color (Ruido={porcentaje_ruido:.0f}%)")
        self.lbl_tit_ifft.config(text=f"Salida IFFT Color: {tipo_fft.upper()} (D0={d0:.0f})")
        self.lbl_tit_acentuado.config(text=tit_ac)

        for arr, lbl in [(esp_norm, self.lbl_img_fft_espectro),
                         (img_ifft_color, self.lbl_img_fft_ifft),
                         (img_color_ruidosa, self.lbl_img_fft_base),
                         (img_ac_color, self.lbl_img_fft_acentuado)]:
            im_p = Image.fromarray(arr)
            im_p.thumbnail((360, 220), Image.Resampling.LANCZOS)
            tk_im = ImageTk.PhotoImage(im_p)
            lbl.config(image=tk_im)
            lbl.image = tk_im

    # =========================================================================
    # PESTAÑA 4: DELIMITACIÓN GEOMÉTRICA (USO EXCLUSIVO DE OPENCV)
    # =========================================================================
    def construir_tab4_delimitacion(self):
        panel_ctrl = tk.Frame(self.tab4, bg="#242424", width=340, padx=10, pady=8)
        panel_ctrl.pack(side=tk.LEFT, fill=tk.Y)
        panel_ctrl.pack_propagate(False)

        tk.Label(panel_ctrl, text="Análisis y Delimitación Geométrica", bg="#242424", fg="#ffd700",
                 font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 2))
        tk.Label(panel_ctrl, text="Uso exclusivo de OpenCV en este módulo:", bg="#242424", fg="#4ec9b0",
                 font=("Arial", 8, "italic")).pack(anchor="w")

        tk.Label(panel_ctrl, text="Método de Segmentación:", bg="#242424", fg="white",
                 font=("Arial", 9, "bold")).pack(anchor="w", pady=(8, 2))
        self.combo_metodo_seg = ttk.Combobox(
            panel_ctrl,
            values=["Otsu Automático (Cualquier Imagen)", "Por Color (Rojo y Amarillo - Caso Sintético)", "Umbral Manual de Intensidad"],
            state="readonly"
        )
        self.combo_metodo_seg.set("Otsu Automático (Cualquier Imagen)")
        self.combo_metodo_seg.pack(fill=tk.X)

        tk.Label(panel_ctrl, text="Umbral Manual (0 a 255):", bg="#242424", fg="lightgray", font=("Arial", 8)).pack(anchor="w", pady=(4, 0))
        self.slider_umbral_man = tk.Scale(panel_ctrl, from_=0, to=255, orient=tk.HORIZONTAL,
                                          bg="#242424", fg="white", highlightthickness=0, troughcolor="#111111")
        self.slider_umbral_man.set(128)
        self.slider_umbral_man.pack(fill=tk.X)

        tk.Button(panel_ctrl, text="⚡ Analizar y Delimitar Regiones", command=self.actualizar_tab4_delimitacion,
                  bg="#0078d4", fg="white", font=("Arial", 9, "bold"), relief=tk.FLAT).pack(fill=tk.X, pady=8)

        tk.Label(panel_ctrl, text="Métricas Extraídas (RegionProps):", bg="#242424", fg="white",
                 font=("Arial", 9, "bold")).pack(anchor="w", pady=(4, 2))
        self.text_props = tk.Text(panel_ctrl, height=10, bg="#111111", fg="#4ec9b0", font=("Consolas", 8))
        self.text_props.pack(fill=tk.X)

        tk.Label(panel_ctrl, text="Código de Cadena de Freeman (8-vecinos):", bg="#242424", fg="#ffd700",
                 font=("Arial", 9, "bold")).pack(anchor="w", pady=(6, 2))
        self.text_freeman = tk.Text(panel_ctrl, height=7, bg="#111111", fg="#dcdcaa", font=("Consolas", 8))
        self.text_freeman.pack(fill=tk.X)

        panel_vis = tk.Frame(self.tab4, bg="#181818", padx=8, pady=8)
        panel_vis.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        panel_vis.columnconfigure(0, weight=2)
        panel_vis.columnconfigure(1, weight=1)
        panel_vis.rowconfigure(0, weight=1)

        f_montaje = tk.Frame(panel_vis, bg="#202020", padx=6, pady=6)
        f_montaje.grid(row=0, column=0, sticky="nsew", padx=4)
        tk.Label(f_montaje, text="Delimitación con Bounding Box, Cuadrado y Contornos", bg="#202020",
                 fg="#4ec9b0", font=("Arial", 10, "bold")).pack()
        self.lbl_img_montaje = tk.Label(f_montaje, bg="#0d0d0d")
        self.lbl_img_montaje.pack(expand=True, fill=tk.BOTH, pady=4)

        f_sub = tk.Frame(panel_vis, bg="#202020", padx=6, pady=6)
        f_sub.grid(row=0, column=1, sticky="nsew", padx=4)
        tk.Label(f_sub, text="Subimagen Binaria 'prop.Image'", bg="#202020",
                 fg="#ffd700", font=("Arial", 10, "bold")).pack()
        self.lbl_img_sub = tk.Label(f_sub, bg="#0d0d0d")
        self.lbl_img_sub.pack(expand=True, fill=tk.BOTH, pady=4)

        self.actualizar_tab4_delimitacion()

    def actualizar_tab4_delimitacion(self):
        r, g, b, gris = separar_canales_rgb(self.img_rgb_original)
        metodo = self.combo_metodo_seg.get()

        if "Otsu" in metodo:
            t_otsu = umbral_otsu_numpy(gris)
            bw_base = (gris < t_otsu).astype(np.uint8) * 255
            bw_solida = imfill_holes(bw_base)
            prop = calcular_propiedades_opencv(bw_solida, etiqueta="Objeto Principal (Otsu)")
            lista_props = [prop] if prop is not None else []

        elif "Color" in metodo:
            r_db, g_db, b_db = r / 255.0, g / 255.0, b / 255.0
            m_roja = (r_db - np.maximum(g_db, b_db) > 0.12) & (r_db > 0.35)
            m_amarilla = (r_db > 0.45) & (g_db > 0.40) & (b_db < 0.40) & (~m_roja)

            bw_a = imfill_holes((m_roja * 255).astype(np.uint8))
            bw_b = imfill_holes((m_amarilla * 255).astype(np.uint8))

            prop_a = calcular_propiedades_opencv(bw_a, etiqueta="Región Roja")
            prop_b = calcular_propiedades_opencv(bw_b, etiqueta="Región Amarilla")
            lista_props = [p for p in [prop_a, prop_b] if p is not None]

        else:
            t_man = int(self.slider_umbral_man.get())
            bw_base = (gris < t_man).astype(np.uint8) * 255
            bw_solida = imfill_holes(bw_base)
            prop = calcular_propiedades_opencv(bw_solida, etiqueta=f"Objeto (T={t_man})")
            lista_props = [prop] if prop is not None else []

        lienzo = self.img_rgb_original.copy()
        alto, ancho, _ = lienzo.shape

        self.text_props.delete("1.0", tk.END)
        self.text_freeman.delete("1.0", tk.END)

        colores = [(0, 255, 60), (0, 220, 255)]

        if not lista_props:
            self.text_props.insert(tk.END, "No se detectaron regiones conexas mayores a 50 px.\nPrueba cambiando el umbral o método.")
            self.lbl_img_montaje.config(image="")
            self.lbl_img_sub.config(image="")
            return

        for idx, p in enumerate(lista_props):
            col = colores[idx % len(colores)]
            cnt = p["contorno"]
            x, y, w, h = p["bbox"]
            cx, cy = p["centroide"]

            # Contorno perimetral cerrado
            cv2.drawContours(lienzo, [cnt], -1, col, 2)
            # Rectángulo Bounding Box
            cv2.rectangle(lienzo, (x, y), (x + w, y + h), col, 2)
            # Cuadrado simétrico lado = max(w, h)
            xq, yq, lado, _ = calcular_cuadrado_simetrico(p["bbox"], p["centroide"], (alto, ancho))
            cv2.rectangle(lienzo, (xq, yq), (xq + lado, yq + lado), (255, 215, 0), 2)
            # Marcador del Centroide
            cv2.circle(lienzo, (int(round(cx)), int(round(cy))), 5, (255, 0, 0), -1)

            # Métricas
            self.text_props.insert(tk.END, f"[{p['etiqueta']}]\n")
            self.text_props.insert(tk.END, f" • Área: {p['area']:.1f} px\n")
            self.text_props.insert(tk.END, f" • Perímetro: {p['perimetro']:.1f} px\n")
            self.text_props.insert(tk.END, f" • Centroide (X, Y): ({cx:.2f}, {cy:.2f})\n")
            self.text_props.insert(tk.END, f" • BoundingBox: [x={x}, y={y}, w={w}, h={h}]\n")
            self.text_props.insert(tk.END, f" • Cuadrado (max w,h): [x={xq}, y={yq}, lado={lado}]\n\n")

            # Freeman
            cadena_str = "".join(str(d) for d in p["freeman"][:60])
            self.text_freeman.insert(tk.END, f"{p['etiqueta']} (Primeros 60):\n{cadena_str}...\n\n")

        im_pil = Image.fromarray(lienzo)
        im_pil.thumbnail((500, 480), Image.Resampling.LANCZOS)
        self.tk_montaje = ImageTk.PhotoImage(im_pil)
        self.lbl_img_montaje.config(image=self.tk_montaje)

        if lista_props:
            sub_im = lista_props[0]["subimagen"]
            sub_pil = Image.fromarray(sub_im)
            sub_pil.thumbnail((250, 250), Image.Resampling.NEAREST)
            self.tk_sub = ImageTk.PhotoImage(sub_pil)
            self.lbl_img_sub.config(image=self.tk_sub)


# =============================================================================
# INICIALIZACIÓN
# =============================================================================

if __name__ == "__main__":
    raiz = tk.Tk()
    app = AplicacionPDIMaestra(raiz)
    raiz.mainloop()