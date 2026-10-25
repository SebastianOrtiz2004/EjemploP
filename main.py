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
# IMPORTACIÓN MODULAR (MÓDULOS 1 AL 5)
# Cada módulo es 100% independiente y reutilizable para el examen
# =============================================================================
from modulo1_canales import formato_uint8, separar_canales_rgb, rgb_a_double
from modulo2_histogramas import (
    calcular_histograma_1d,
    calcular_pdf_normalizada,
    aplicar_contraste_lineal,
    redistribuir_canal_numpy,
)
from modulo3_espacial import (
    agregar_padding_ceros,
    calcular_matriz_gaussiana,
    inyectar_ruido_sal_pimienta,
    convolucion_con_salto_manual,
    filtro_mediana_manual,
)
from modulo4_frecuencia import acentuado_espacial, filtrado_frecuencia_fft
from modulo5_geometria import (
    umbral_otsu_numpy,
    imfill_holes,
    calcular_propiedades_opencv,
    calcular_cuadrado_simetrico,
)


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