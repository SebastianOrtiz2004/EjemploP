"""
=============================================================================
SISTEMA INTEGRADO DE PROCESAMIENTO DIGITAL DE IMÁGENES (PDI) - EXAMEN PRÁCTICO
=============================================================================
Control estricto de librerías:
- Pestañas 1, 2, 3 y 4: Implementación matemática con NumPy, PIL y Matplotlib.
  (Cero OpenCV en: carga, RGB, gris NTSC, histogramas, convolución, saltos,
   FFT2, filtros de borde, Sobel, Kirsch, LoG, Frei-Chen y Relajación).
- Pestaña 5: Uso EXCLUSIVO de OpenCV (cv2) para delimitación, imfill, momentos,
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

# OpenCV PERMITIDO ÚNICAMENTE para la Pestaña 5 (Análisis Geométrico y Delimitación)
import cv2

# Importaciones modulares
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
from modulo_bordes import (
    operador_roberts_2x2,
    operador_prewitt_3x3,
    operador_sobel_3x3,
    operador_kirsch_4,
    operador_log_dog,
    operador_frei_chen_9,
    relajacion_bordes_iterativa,
)
from modulo5_geometria import (
    umbral_otsu_numpy,
    imfill_holes,
    calcular_propiedades_opencv,
    calcular_cuadrado_simetrico,
)


class AplicacionPDIMaestra:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("Sistema de Procesamiento Digital de Imágenes - Examen Práctico")
        self.ventana.geometry("1440x880")
        self.ventana.minsize(1280, 720)
        self.ventana.configure(bg="#181818")

        self.img_rgb_original = None
        self.cargar_imagen_defecto()

        estilo = ttk.Style()
        try:
            estilo.theme_use("clam")
        except Exception:
            pass
        estilo.configure("TNotebook", background="#181818", borderwidth=0)
        estilo.configure("TNotebook.Tab", background="#2d2d30", foreground="#cccccc",
                        padding=[14, 6], font=("Arial", 9, "bold"))
        estilo.map("TNotebook.Tab", background=[("selected", "#0078d4")], foreground=[("selected", "#ffffff")])

        # Barra superior de carga común
        top_bar = tk.Frame(self.ventana, bg="#242424", padx=12, pady=6)
        top_bar.pack(fill=tk.X)

        tk.Button(top_bar, text="📁 Cargar Imagen (Cualquier Formato)", command=self.dialogo_cargar_imagen,
                  bg="#0078d4", fg="white", font=("Arial", 9, "bold"), relief=tk.FLAT, padx=10).pack(side=tk.LEFT)

        self.lbl_info_archivo = tk.Label(top_bar, text="Imagen: Sintética Calibrada (320x420)",
                                         bg="#242424", fg="#4ec9b0", font=("Arial", 9))
        self.lbl_info_archivo.pack(side=tk.LEFT, padx=15)

        # Contenedor de 5 Pestañas
        self.notebook = ttk.Notebook(self.ventana)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.tab1 = tk.Frame(self.notebook, bg="#181818")
        self.tab2 = tk.Frame(self.notebook, bg="#181818")
        self.tab3 = tk.Frame(self.notebook, bg="#181818")
        self.tab4_bordes = tk.Frame(self.notebook, bg="#181818")
        self.tab5_geom = tk.Frame(self.notebook, bg="#181818")

        self.notebook.add(self.tab1, text="  1. Descomposición RGB y 3 Histogramas  ")
        self.notebook.add(self.tab2, text="  2. Filtrado Espacial y Stride  ")
        self.notebook.add(self.tab3, text="  3. Frecuencia FFT2 y Acentuado  ")
        self.notebook.add(self.tab4_bordes, text="  4. Bordes, Gradientes y Relajación  ")
        self.notebook.add(self.tab5_geom, text="  5. Delimitación y Freeman (OpenCV)  ")

        # Construcción de componentes
        self.construir_tab1_canales_e_histogramas()
        self.construir_tab2_espacial()
        self.construir_tab3_frecuencia()
        self.construir_tab4_bordes_y_relajacion()
        self.construir_tab5_delimitacion()

    def cargar_imagen_defecto(self):
        alto, ancho = 320, 420
        self.img_rgb_original = np.full((alto, ancho, 3), 240, dtype=np.uint8)
        self.img_rgb_original[60:240, 50:180] = [215, 35, 35]
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
                self.actualizar_tab1()
                self.actualizar_tab2_preview()
                self.actualizar_tab3_ffts()
                self.actualizar_tab4_bordes()
                self.actualizar_tab5_delimitacion()
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

        f_r = tk.LabelFrame(panel_ctrl, text="🔴 Canal Rojo", bg="#202020", fg="#ff6b6b", font=("Arial", 8, "bold"), padx=4, pady=2)
        f_r.pack(fill=tk.X, pady=2)
        self.slider_c_r = tk.Scale(f_r, from_=0.2, to=3.0, resolution=0.05, orient=tk.HORIZONTAL, bg="#202020", fg="white", highlightthickness=0, troughcolor="#111111")
        self.slider_c_r.set(1.0)
        self.slider_c_r.pack(fill=tk.X)
        self.slider_c_r.bind("<ButtonRelease-1>", lambda _: self.actualizar_tab1())

        self.slider_red_r = tk.Scale(f_r, from_=0, to=100, resolution=5, orient=tk.HORIZONTAL, bg="#202020", fg="#ff6b6b", highlightthickness=0, troughcolor="#111111")
        self.slider_red_r.set(0)
        self.slider_red_r.pack(fill=tk.X)
        self.slider_red_r.bind("<ButtonRelease-1>", lambda _: self.actualizar_tab1())

        f_g = tk.LabelFrame(panel_ctrl, text="🟢 Canal Verde", bg="#202020", fg="#51cf66", font=("Arial", 8, "bold"), padx=4, pady=2)
        f_g.pack(fill=tk.X, pady=2)
        self.slider_c_g = tk.Scale(f_g, from_=0.2, to=3.0, resolution=0.05, orient=tk.HORIZONTAL, bg="#202020", fg="white", highlightthickness=0, troughcolor="#111111")
        self.slider_c_g.set(1.0)
        self.slider_c_g.pack(fill=tk.X)
        self.slider_c_g.bind("<ButtonRelease-1>", lambda _: self.actualizar_tab1())

        self.slider_red_g = tk.Scale(f_g, from_=0, to=100, resolution=5, orient=tk.HORIZONTAL, bg="#202020", fg="#51cf66", highlightthickness=0, troughcolor="#111111")
        self.slider_red_g.set(0)
        self.slider_red_g.pack(fill=tk.X)
        self.slider_red_g.bind("<ButtonRelease-1>", lambda _: self.actualizar_tab1())

        f_b = tk.LabelFrame(panel_ctrl, text="🔵 Canal Azul", bg="#202020", fg="#339af0", font=("Arial", 8, "bold"), padx=4, pady=2)
        f_b.pack(fill=tk.X, pady=2)
        self.slider_c_b = tk.Scale(f_b, from_=0.2, to=3.0, resolution=0.05, orient=tk.HORIZONTAL, bg="#202020", fg="white", highlightthickness=0, troughcolor="#111111")
        self.slider_c_b.set(1.0)
        self.slider_c_b.pack(fill=tk.X)
        self.slider_c_b.bind("<ButtonRelease-1>", lambda _: self.actualizar_tab1())

        self.slider_red_b = tk.Scale(f_b, from_=0, to=100, resolution=5, orient=tk.HORIZONTAL, bg="#202020", fg="#339af0", highlightthickness=0, troughcolor="#111111")
        self.slider_red_b.set(0)
        self.slider_red_b.pack(fill=tk.X)
        self.slider_red_b.bind("<ButtonRelease-1>", lambda _: self.actualizar_tab1())

        tk.Button(panel_ctrl, text="↺ Restaurar Canales", command=self.reset_tab1_valores, bg="#3e3e42", fg="white", font=("Arial", 8, "bold"), relief=tk.FLAT).pack(fill=tk.X, pady=4)

        panel_derecho = tk.Frame(self.tab1, bg="#181818", padx=4, pady=2)
        panel_derecho.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        frame_galeria = tk.Frame(panel_derecho, bg="#1c1c1c", padx=3, pady=3)
        frame_galeria.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 2))
        for col_idx in range(5):
            frame_galeria.columnconfigure(col_idx, weight=1, uniform="galeria_col")
        frame_galeria.rowconfigure(0, weight=1)

        tarjetas = [
            ("1. Original (RGB)", "white", "lbl_v_orig"),
            ("2. Resultante (RGB)", "#4ec9b0", "lbl_v_res"),
            ("3. Canal Rojo (Gris)", "#ff6b6b", "lbl_v_r"),
            ("4. Canal Verde (Gris)", "#51cf66", "lbl_v_g"),
            ("5. Canal Azul (Gris)", "#339af0", "lbl_v_b")
        ]
        for col_idx, (tit, col, attr) in enumerate(tarjetas):
            t_frame = tk.Frame(frame_galeria, bg="#181818", padx=2, pady=2)
            t_frame.grid(row=0, column=col_idx, sticky="nsew", padx=2, pady=2)
            tk.Label(t_frame, text=tit, bg="#181818", fg=col, font=("Arial", 8, "bold")).pack()
            lbl = tk.Label(t_frame, bg="#0d0d0d")
            lbl.pack(expand=True, fill=tk.BOTH, pady=2)
            setattr(self, attr, lbl)

        frame_hist = tk.Frame(panel_derecho, bg="#181818", height=240)
        frame_hist.pack(side=tk.BOTTOM, fill=tk.X, pady=(2, 0))
        frame_hist.pack_propagate(False)

        self.fig_hist, self.axes_hist = plt.subplots(1, 3, figsize=(10, 2.2))
        self.fig_hist.patch.set_facecolor("#181818")
        self.canvas_hist = FigureCanvasTkAgg(self.fig_hist, master=frame_hist)
        self.canvas_hist.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.actualizar_tab1()

    def reset_tab1_valores(self):
        for s in [self.slider_c_r, self.slider_c_g, self.slider_c_b]:
            s.set(1.0)
        for s in [self.slider_red_r, self.slider_red_g, self.slider_red_b]:
            s.set(0)
        self.actualizar_tab1()

    def actualizar_tab1(self):
        c_r, red_r = float(self.slider_c_r.get()), float(self.slider_red_r.get())
        c_g, red_g = float(self.slider_c_g.get()), float(self.slider_red_g.get())
        c_b, red_b = float(self.slider_c_b.get()), float(self.slider_red_b.get())

        r, g, b, _ = separar_canales_rgb(self.img_rgb_original)
        r_mod = redistribuir_canal_numpy(aplicar_contraste_lineal(r, c_r), red_r)
        g_mod = redistribuir_canal_numpy(aplicar_contraste_lineal(g, c_g), red_g)
        b_mod = redistribuir_canal_numpy(aplicar_contraste_lineal(b, c_b), red_b)
        img_res = np.stack([r_mod, g_mod, b_mod], axis=-1)

        for mat, modo, lbl in [
            (self.img_rgb_original, "RGB", self.lbl_v_orig),
            (img_res, "RGB", self.lbl_v_res),
            (r_mod, "L", self.lbl_v_r),
            (g_mod, "L", self.lbl_v_g),
            (b_mod, "L", self.lbl_v_b)
        ]:
            p_img = Image.fromarray(mat, mode=modo)
            p_img.thumbnail((190, 150), Image.Resampling.LANCZOS)
            tk_im = ImageTk.PhotoImage(p_img)
            lbl.config(image=tk_im)
            lbl.image = tk_im

        h_r_orig, h_r_mod = calcular_histograma_1d(r), calcular_histograma_1d(r_mod)
        h_g_orig, h_g_mod = calcular_histograma_1d(g), calcular_histograma_1d(g_mod)
        h_b_orig, h_b_mod = calcular_histograma_1d(b), calcular_histograma_1d(b_mod)

        canales_info = [
            (self.axes_hist[0], "#ff4444", "Canal Rojo", h_r_orig, h_r_mod),
            (self.axes_hist[1], "#00cc66", "Canal Verde", h_g_orig, h_g_mod),
            (self.axes_hist[2], "#3399ff", "Canal Azul", h_b_orig, h_b_mod)
        ]
        for ax, col, nom, ho, hm in canales_info:
            ax.clear()
            ax.set_facecolor("#111111")
            ax.tick_params(colors="gray", labelsize=7)
            ax.plot(range(256), ho, color="#777777", linestyle="--", linewidth=1.0, label="Orig")
            ax.bar(range(256), hm, color=col, width=1.0, alpha=0.85, label="Mod")
            ax.set_title(f"Histograma {nom}", color="white", fontsize=8, weight="bold")
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

        tk.Label(panel_ctrl, text="1. Ruido Sal y Pimienta:", bg="#242424", fg="#ffd700", font=("Arial", 9, "bold")).pack(anchor="w")
        self.slider_ruido = tk.Scale(panel_ctrl, from_=0.0, to=0.5, resolution=0.02, orient=tk.HORIZONTAL, bg="#242424", fg="white", highlightthickness=0, troughcolor="#111111")
        self.slider_ruido.set(0.05)
        self.slider_ruido.pack(fill=tk.X, pady=(0, 4))

        tk.Label(panel_ctrl, text="2. Técnica de Suavizado:", bg="#242424", fg="white", font=("Arial", 9, "bold")).pack(anchor="w", pady=(4, 2))
        self.combo_filtro_espacial = ttk.Combobox(panel_ctrl, values=["Lineal (Gaussiana Calculada)", "Lineal (Media / Promedio)", "No Lineal (Mediana)"], state="readonly", font=("Arial", 9))
        self.combo_filtro_espacial.set("Lineal (Gaussiana Calculada)")
        self.combo_filtro_espacial.pack(fill=tk.X)
        self.combo_filtro_espacial.bind("<<ComboboxSelected>>", lambda _: self.actualizar_matriz_text())

        f_tam = tk.Frame(panel_ctrl, bg="#242424")
        f_tam.pack(fill=tk.X, pady=6)
        tk.Label(f_tam, text="Orden KxK:", bg="#242424", fg="white", font=("Arial", 9, "bold")).pack(side=tk.LEFT)
        self.entry_tam_k = tk.Entry(f_tam, width=4, justify="center", font=("Arial", 9, "bold"))
        self.entry_tam_k.insert(0, "3")
        self.entry_tam_k.pack(side=tk.LEFT, padx=6)
        tk.Button(f_tam, text="Recalcular", command=self.actualizar_matriz_text, bg="#3e3e42", fg="white", font=("Arial", 8, "bold"), relief=tk.FLAT).pack(side=tk.LEFT)

        tk.Label(panel_ctrl, text="Máscara W(x,y):", bg="#242424", fg="#4ec9b0", font=("Arial", 8, "bold")).pack(anchor="w", pady=(4, 1))
        self.text_matriz = tk.Text(panel_ctrl, height=5, bg="#111111", fg="#ffd700", font=("Consolas", 8), relief=tk.SOLID, bd=1)
        self.text_matriz.pack(fill=tk.X)

        f_saltos = tk.Frame(panel_ctrl, bg="#242424")
        f_saltos.pack(fill=tk.X, pady=6)
        tk.Label(f_saltos, text="Salto F:", bg="#242424", fg="lightgray", font=("Arial", 8)).grid(row=0, column=0)
        self.entry_salto_f = tk.Entry(f_saltos, width=4, justify="center")
        self.entry_salto_f.insert(0, "1")
        self.entry_salto_f.grid(row=0, column=1, padx=4)
        tk.Label(f_saltos, text="Salto C:", bg="#242424", fg="lightgray", font=("Arial", 8)).grid(row=0, column=2)
        self.entry_salto_c = tk.Entry(f_saltos, width=4, justify="center")
        self.entry_salto_c.insert(0, "1")
        self.entry_salto_c.grid(row=0, column=3, padx=4)

        self.var_padding = tk.BooleanVar(value=True)
        tk.Checkbutton(panel_ctrl, text="Aplicar Padding de Ceros", variable=self.var_padding, bg="#242424", fg="white", selectcolor="#111111", font=("Arial", 8)).pack(anchor="w")

        tk.Button(panel_ctrl, text="⚡ Ejecutar Convolución y Stride", command=self.ejecutar_filtro_espacial, bg="#107c41", fg="white", font=("Arial", 9, "bold"), relief=tk.FLAT).pack(fill=tk.X, pady=8)

        panel_vis = tk.Frame(self.tab2, bg="#181818", padx=8, pady=8)
        panel_vis.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        panel_vis.columnconfigure(0, weight=1)
        panel_vis.columnconfigure(1, weight=1)
        panel_vis.rowconfigure(0, weight=1)

        f_izq = tk.Frame(panel_vis, bg="#202020", padx=4, pady=4)
        f_izq.grid(row=0, column=0, sticky="nsew", padx=4)
        tk.Label(f_izq, text="Entrada con Ruido", bg="#202020", fg="white", font=("Arial", 9, "bold")).pack()
        self.lbl_dim_esp_orig = tk.Label(f_izq, text="", bg="#202020", fg="gray", font=("Arial", 8))
        self.lbl_dim_esp_orig.pack()
        self.lbl_img_esp_orig = tk.Label(f_izq, bg="#0d0d0d")
        self.lbl_img_esp_orig.pack(expand=True, fill=tk.BOTH)

        f_der = tk.Frame(panel_vis, bg="#202020", padx=4, pady=4)
        f_der.grid(row=0, column=1, sticky="nsew", padx=4)
        tk.Label(f_der, text="Filtrada y Reducida", bg="#202020", fg="#4ec9b0", font=("Arial", 9, "bold")).pack()
        self.lbl_dim_esp_res = tk.Label(f_der, text="", bg="#202020", fg="#4ec9b0", font=("Arial", 8, "bold"))
        self.lbl_dim_esp_res.pack()
        self.lbl_img_esp_res = tk.Label(f_der, bg="#0d0d0d")
        self.lbl_img_esp_res.pack(expand=True, fill=tk.BOTH)

        self.actualizar_matriz_text()
        self.actualizar_tab2_preview()

    def actualizar_matriz_text(self):
        try:
            k = int(self.entry_tam_k.get())
            if k < 3 or k % 2 == 0:
                raise ValueError
        except ValueError:
            self.entry_tam_k.delete(0, tk.END)
            self.entry_tam_k.insert(0, "3")
            k = 3

        tipo = self.combo_filtro_espacial.get()
        self.text_matriz.delete("1.0", tk.END)
        if "Gaussiana" in tipo:
            w = calcular_matriz_gaussiana(k, sigma=1.0)
            texto = f"Gaussiana analítica (k={k}, suma=1.0):\n" + "\n".join(["[" + " ".join([f"{v:6.4f}" for v in row]) + "]" for row in w])
        elif "Media" in tipo:
            w = np.ones((k, k), dtype=np.float64) / float(k * k)
            texto = f"Media uniforme (k={k}, peso=1/{k*k}):\n" + "\n".join(["[" + " ".join([f"{v:6.4f}" for v in row]) + "]" for row in w])
        else:
            texto = f"Filtro No Lineal ({tipo}):\nOrdenamiento local de vecindad de {k}x{k}."
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
            r_out = filtro_mediana_manual(r, k, salto_f, salto_c)
            g_out = filtro_mediana_manual(g, k, salto_f, salto_c)
            b_out = filtro_mediana_manual(b, k, salto_f, salto_c)

        img_red = np.stack([r_out, g_out, b_out], axis=-1)
        h_r, w_r, _ = img_red.shape
        self.lbl_dim_esp_res.config(text=f"{h_r}x{w_r} px (Salto {salto_f}x{salto_c})")
        im_pil = Image.fromarray(img_red)
        im_pil.thumbnail((440, 440), Image.Resampling.LANCZOS)
        self.tk_reducida = ImageTk.PhotoImage(im_pil)
        self.lbl_img_esp_res.config(image=self.tk_reducida)

    # =========================================================================
    # PESTAÑA 3: FRECUENCIA (FFT 2D) Y ACENTUADO ESPACIAL
    # =========================================================================
    def construir_tab3_frecuencia(self):
        panel_ctrl = tk.Frame(self.tab3, bg="#242424", width=300, padx=10, pady=8)
        panel_ctrl.pack(side=tk.LEFT, fill=tk.Y)
        panel_ctrl.pack_propagate(False)

        tk.Label(panel_ctrl, text="Filtrado en Frecuencia (FFT)", bg="#242424", fg="#ffd700", font=("Arial", 10, "bold")).pack(anchor="w")

        tk.Label(panel_ctrl, text="Ruido Entrada (%):", bg="#242424", fg="white", font=("Arial", 8, "bold")).pack(anchor="w", pady=(6, 1))
        self.slider_ruido_fft = tk.Scale(panel_ctrl, from_=0, to=100, orient=tk.HORIZONTAL, bg="#242424", fg="white", highlightthickness=0, troughcolor="#111111")
        self.slider_ruido_fft.set(0)
        self.slider_ruido_fft.pack(fill=tk.X)

        tk.Label(panel_ctrl, text="Tipo de Filtro:", bg="#242424", fg="white", font=("Arial", 8, "bold")).pack(anchor="w", pady=(6, 1))
        self.combo_tipo_fft = ttk.Combobox(panel_ctrl, values=["pasa_bajos", "pasa_altos"], state="readonly")
        self.combo_tipo_fft.set("pasa_bajos")
        self.combo_tipo_fft.pack(fill=tk.X)

        tk.Label(panel_ctrl, text="Radio D0 (px):", bg="#242424", fg="white", font=("Arial", 8, "bold")).pack(anchor="w", pady=(6, 1))
        self.slider_d0 = tk.Scale(panel_ctrl, from_=5, to=150, resolution=5, orient=tk.HORIZONTAL, bg="#242424", fg="white", highlightthickness=0, troughcolor="#111111")
        self.slider_d0.set(40)
        self.slider_d0.pack(fill=tk.X)

        tk.Label(panel_ctrl, text="Modo de Espectro:", bg="#242424", fg="#4ec9b0", font=("Arial", 8, "bold")).pack(anchor="w", pady=(6, 1))
        self.combo_modo_espectro = ttk.Combobox(panel_ctrl, values=["logaritmico", "lineal", "fase", "mascara_filtro_H"], state="readonly")
        self.combo_modo_espectro.set("logaritmico")
        self.combo_modo_espectro.pack(fill=tk.X)

        self.var_aplicar_acentuado = tk.BooleanVar(value=False)
        tk.Checkbutton(panel_ctrl, text="Aplicar Acentuado Espacial", variable=self.var_aplicar_acentuado, bg="#242424", fg="white", selectcolor="#111111", font=("Arial", 8, "bold")).pack(anchor="w", pady=(8, 2))

        self.combo_modo_acentuado = ttk.Combobox(panel_ctrl, values=["high_boost", "solo_bordes"], state="readonly")
        self.combo_modo_acentuado.set("high_boost")
        self.combo_modo_acentuado.pack(fill=tk.X)

        self.slider_factor_a = tk.Scale(panel_ctrl, from_=1.0, to=2.5, resolution=0.1, orient=tk.HORIZONTAL, bg="#242424", fg="white", highlightthickness=0, troughcolor="#111111")
        self.slider_factor_a.set(1.2)
        self.slider_factor_a.pack(fill=tk.X)

        tk.Button(panel_ctrl, text="⚡ Ejecutar FFT + Acentuado", command=self.actualizar_tab3_ffts, bg="#107c41", fg="white", font=("Arial", 9, "bold"), relief=tk.FLAT).pack(fill=tk.X, pady=10)

        panel_vis = tk.Frame(self.tab3, bg="#181818", padx=6, pady=6)
        panel_vis.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        panel_vis.columnconfigure(0, weight=1)
        panel_vis.columnconfigure(1, weight=1)
        panel_vis.rowconfigure(0, weight=1)
        panel_vis.rowconfigure(1, weight=1)

        paneles = [
            (0, 0, "lbl_tit_espectro", "Espectro / Máscara", "lbl_img_fft_espectro"),
            (0, 1, "lbl_tit_ifft", "Salida IFFT 2D", "lbl_img_fft_ifft"),
            (1, 0, "lbl_tit_entrada", "Entrada Color", "lbl_img_fft_base"),
            (1, 1, "lbl_tit_acentuado", "Resultado Acentuado", "lbl_img_fft_acentuado")
        ]
        for r_idx, c_idx, tit_attr, tit_txt, img_attr in paneles:
            f = tk.Frame(panel_vis, bg="#202020", padx=4, pady=4)
            f.grid(row=r_idx, column=c_idx, sticky="nsew", padx=3, pady=3)
            l_tit = tk.Label(f, text=tit_txt, bg="#202020", fg="white", font=("Arial", 9, "bold"))
            l_tit.pack()
            setattr(self, tit_attr, l_tit)
            l_img = tk.Label(f, bg="#0d0d0d")
            l_img.pack(expand=True, fill=tk.BOTH)
            setattr(self, img_attr, l_img)

        self.actualizar_tab3_ffts()

    def actualizar_tab3_ffts(self):
        tipo_fft = self.combo_tipo_fft.get()
        d0 = float(self.slider_d0.get())
        dens_ruido = float(self.slider_ruido_fft.get()) / 100.0
        modo_esp = self.combo_modo_espectro.get()

        img_base = inyectar_ruido_sal_pimienta(self.img_rgb_original, dens_ruido)
        img_ifft_color, espectro = filtrado_frecuencia_fft(img_base, tipo=tipo_fft, d0=d0, modo_espectro=modo_esp)

        min_e, max_e = np.min(espectro), np.max(espectro)
        esp_norm = (255.0 * (espectro - min_e) / (max_e - min_e + 1e-6)).astype(np.uint8) if max_e > min_e else np.zeros_like(espectro, dtype=np.uint8)

        if self.var_aplicar_acentuado.get():
            modo_ac = self.combo_modo_acentuado.get()
            factor_a = float(self.slider_factor_a.get())
            img_ac = acentuado_espacial(img_ifft_color, modo=modo_ac, factor_a=factor_a)
            self.lbl_tit_acentuado.config(text=f"Acentuado: {modo_ac.upper()} (A={factor_a:.1f})")
        else:
            img_ac = img_ifft_color
            self.lbl_tit_acentuado.config(text="Acentuado Inactivo")

        self.lbl_tit_espectro.config(text=f"Espectro: {modo_esp.upper()}")
        self.lbl_tit_entrada.config(text=f"Entrada (Ruido={dens_ruido*100:.0f}%)")
        self.lbl_tit_ifft.config(text=f"Salida IFFT ({tipo_fft.upper()} D0={d0:.0f})")

        for arr, lbl in [(esp_norm, self.lbl_img_fft_espectro), (img_ifft_color, self.lbl_img_fft_ifft), (img_base, self.lbl_img_fft_base), (img_ac, self.lbl_img_fft_acentuado)]:
            im_p = Image.fromarray(arr)
            im_p.thumbnail((360, 220), Image.Resampling.LANCZOS)
            tk_im = ImageTk.PhotoImage(im_p)
            lbl.config(image=tk_im)
            lbl.image = tk_im

    # =========================================================================
    # PESTAÑA 4: DETECCIÓN DE BORDES, GRADIENTES, KIRSCH, LOG, FREI-CHEN Y RELAJACIÓN
    # =========================================================================
    def construir_tab4_bordes_y_relajacion(self):
        panel_ctrl = tk.Frame(self.tab4_bordes, bg="#242424", width=340, padx=10, pady=8)
        panel_ctrl.pack(side=tk.LEFT, fill=tk.Y)
        panel_ctrl.pack_propagate(False)

        tk.Label(panel_ctrl, text="Detección de Bordes y Derivadas", bg="#242424", fg="#ffd700", font=("Arial", 10, "bold")).pack(anchor="w")

        tk.Label(panel_ctrl, text="1. Operador / Técnica:", bg="#242424", fg="white", font=("Arial", 9, "bold")).pack(anchor="w", pady=(6, 1))
        self.combo_operador_borde = ttk.Combobox(
            panel_ctrl,
            values=[
                "Sobel (3x3)",
                "Prewitt (3x3)",
                "Roberts (2x2)",
                "Kirsch (4 Direcciones: 0°, 45°, 90°, 135°)",
                "Laplaciano de Gaussiana (DoG + Cruce por Cero)",
                "Máscaras Ortogonales Frei-Chen (Orillas vs Líneas)",
                "Relajación Contextual Iterativa"
            ],
            state="readonly",
            font=("Arial", 8)
        )
        self.combo_operador_borde.set("Sobel (3x3)")
        self.combo_operador_borde.pack(fill=tk.X)
        self.combo_operador_borde.bind("<<ComboboxSelected>>", self.on_cambio_operador_borde)

        # Controles para DoG / LoG
        self.frame_dog_ctrl = tk.LabelFrame(panel_ctrl, text="Parámetros DoG (LoG)", bg="#202020", fg="#4ec9b0", font=("Arial", 8, "bold"), padx=4, pady=2)
        tk.Label(self.frame_dog_ctrl, text="Sigma 1:", bg="#202020", fg="white", font=("Arial", 8)).grid(row=0, column=0)
        self.entry_s1 = tk.Entry(self.frame_dog_ctrl, width=4, justify="center")
        self.entry_s1.insert(0, "0.5")
        self.entry_s1.grid(row=0, column=1, padx=2)
        tk.Label(self.frame_dog_ctrl, text="Sigma 2:", bg="#202020", fg="white", font=("Arial", 8)).grid(row=0, column=2)
        self.entry_s2 = tk.Entry(self.frame_dog_ctrl, width=4, justify="center")
        self.entry_s2.insert(0, "0.8")
        self.entry_s2.grid(row=0, column=3, padx=2)

        # Control para Relajación
        self.frame_relaj_ctrl = tk.LabelFrame(panel_ctrl, text="Parámetros Relajación", bg="#202020", fg="#4ec9b0", font=("Arial", 8, "bold"), padx=4, pady=2)
        tk.Label(self.frame_relaj_ctrl, text="Iteraciones:", bg="#202020", fg="white", font=("Arial", 8)).pack(side=tk.LEFT)
        self.slider_iter_relaj = tk.Scale(self.frame_relaj_ctrl, from_=1, to=10, orient=tk.HORIZONTAL, bg="#202020", fg="white", highlightthickness=0, troughcolor="#111111")
        self.slider_iter_relaj.set(3)
        self.slider_iter_relaj.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Button(panel_ctrl, text="⚡ Ejecutar Detección de Bordes", command=self.actualizar_tab4_bordes, bg="#107c41", fg="white", font=("Arial", 9, "bold"), relief=tk.FLAT).pack(fill=tk.X, pady=8)

        tk.Label(panel_ctrl, text="Fundamento Teórico y Máscara:", bg="#242424", fg="#4ec9b0", font=("Arial", 8, "bold")).pack(anchor="w", pady=(4, 1))
        self.text_teoria_bordes = tk.Text(panel_ctrl, height=14, bg="#111111", fg="#ffd700", font=("Consolas", 8), relief=tk.SOLID, bd=1)
        self.text_teoria_bordes.pack(fill=tk.BOTH, expand=True)

        # 4 Paneles de visualización
        panel_vis = tk.Frame(self.tab4_bordes, bg="#181818", padx=6, pady=6)
        panel_vis.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        panel_vis.columnconfigure(0, weight=1)
        panel_vis.columnconfigure(1, weight=1)
        panel_vis.rowconfigure(0, weight=1)
        panel_vis.rowconfigure(1, weight=1)

        b_paneles = [
            (0, 0, "lbl_tit_b1", "Imagen Entrada", "lbl_img_b1"),
            (0, 1, "lbl_tit_b2", "Componente Gx / R0", "lbl_img_b2"),
            (1, 0, "lbl_tit_b3", "Componente Gy / Dirección", "lbl_img_b3"),
            (1, 1, "lbl_tit_b4", "Magnitud Final / Resultado", "lbl_img_b4")
        ]
        for r_idx, c_idx, tit_attr, tit_txt, img_attr in b_paneles:
            f = tk.Frame(panel_vis, bg="#202020", padx=4, pady=4)
            f.grid(row=r_idx, column=c_idx, sticky="nsew", padx=3, pady=3)
            l_tit = tk.Label(f, text=tit_txt, bg="#202020", fg="white", font=("Arial", 9, "bold"))
            l_tit.pack()
            setattr(self, tit_attr, l_tit)
            l_img = tk.Label(f, bg="#0d0d0d")
            l_img.pack(expand=True, fill=tk.BOTH)
            setattr(self, img_attr, l_img)

        self.on_cambio_operador_borde()
        self.actualizar_tab4_bordes()

    def on_cambio_operador_borde(self, event=None):
        op = self.combo_operador_borde.get()
        self.frame_dog_ctrl.pack_forget()
        self.frame_relaj_ctrl.pack_forget()

        if "DoG" in op or "Laplaciano" in op:
            self.frame_dog_ctrl.pack(fill=tk.X, pady=4)
        elif "Relajación" in op:
            self.frame_relaj_ctrl.pack(fill=tk.X, pady=4)

    def actualizar_tab4_bordes(self):
        _, _, _, gris = separar_canales_rgb(self.img_rgb_original)
        op = self.combo_operador_borde.get()
        self.text_teoria_bordes.delete("1.0", tk.END)

        def normalizar_u8(mat):
            mn, mx = np.min(mat), np.max(mat)
            if mx > mn:
                return ((mat - mn) / (mx - mn) * 255.0).astype(np.uint8)
            return np.zeros_like(mat, dtype=np.uint8)

        if "Roberts" in op:
            gx, gy, mag = operador_roberts_2x2(gris)
            self.lbl_tit_b1.config(text="Imagen Original (Gris NTSC)")
            self.lbl_tit_b2.config(text="Roberts Gx (Diagonal 1)")
            self.lbl_tit_b3.config(text="Roberts Gy (Diagonal 2)")
            self.lbl_tit_b4.config(text="Magnitud |Gx| + |Gy|")
            arr_b1, arr_b2, arr_b3, arr_b4 = gris, normalizar_u8(gx), normalizar_u8(gy), normalizar_u8(mag)
            teoria = (
                "OPERADOR DE ROBERTS (2x2)\n"
                "• Diferencias cruzadas diagonales:\n"
                "  Gx = [[1,  0], [0, -1]]\n"
                "  Gy = [[0,  1], [-1, 0]]\n"
                "• Magnitud: |Gx| + |Gy|\n"
                "• Muy rápido pero vulnerable al ruido por falta de promediado."
            )

        elif "Prewitt" in op:
            gx, gy, mag, ang = operador_prewitt_3x3(gris)
            self.lbl_tit_b1.config(text="Imagen Original (Gris)")
            self.lbl_tit_b2.config(text="Prewitt Gx (Vertical)")
            self.lbl_tit_b3.config(text="Prewitt Gy (Horizontal)")
            self.lbl_tit_b4.config(text="Magnitud |Gx| + |Gy|")
            arr_b1, arr_b2, arr_b3, arr_b4 = gris, normalizar_u8(gx), normalizar_u8(gy), normalizar_u8(mag)
            teoria = (
                "OPERADOR DE PREWITT (3x3)\n"
                "• Derivadas con promedio simple:\n"
                "  Gx = [[-1,-1,-1],[0,0,0],[1,1,1]]\n"
                "  Gy = [[-1,0,1],[-1,0,1],[-1,0,1]]\n"
                "• Dirección de orilla: phi = atan(Gy/Gx)\n"
                "• Pondera por igual píxeles alineados y esquinas."
            )

        elif "Sobel" in op:
            gx, gy, mag, ang = operador_sobel_3x3(gris)
            self.lbl_tit_b1.config(text="Imagen Original (Gris)")
            self.lbl_tit_b2.config(text="Sobel Gx (Gradiente Vertical)")
            self.lbl_tit_b3.config(text="Sobel Gy (Gradiente Horizontal)")
            self.lbl_tit_b4.config(text="Magnitud sqrt(Gx^2 + Gy^2)")
            arr_b1, arr_b2, arr_b3, arr_b4 = gris, normalizar_u8(gx), normalizar_u8(gy), normalizar_u8(mag)
            teoria = (
                "OPERADOR DE SOBEL (3x3)\n"
                "• Diferenciación + Suavizado Gaussiano:\n"
                "  Sx = [[-1,-2,-1],[0,0,0],[1,2,1]]\n"
                "  Sy = [[-1,0,1],[-2,0,2],[-1,0,1]]\n"
                "• Descomposición: D=[-1,0,1], G=[1,2,1]^T\n"
                "• Reduce notablemente el ruido antes de derivar."
            )

        elif "Kirsch" in op:
            r0, r45, r90, r135, mag_max, angulos = operador_kirsch_4(gris)
            self.lbl_tit_b1.config(text="Plantilla 0° (Horizontal)")
            self.lbl_tit_b2.config(text="Plantilla 45° (Diagonal)")
            self.lbl_tit_b3.config(text="Plantilla 90° (Vertical)")
            self.lbl_tit_b4.config(text="Máxima Respuesta Kirsch")
            arr_b1, arr_b2, arr_b3, arr_b4 = normalizar_u8(r0), normalizar_u8(r45), normalizar_u8(r90), normalizar_u8(mag_max)
            teoria = (
                "PLANTILLAS DE KIRSCH (4 DIRECCIONES)\n"
                "• Máscaras direccionales de 3x3 a 0°, 45°, 90°, 135°:\n"
                "  K0   = [[-1,-1,-1],[0,0,0],[1,1,1]]\n"
                "  K45  = [[-1,-1,0],[-1,0,1],[0,1,1]]\n"
                "  K90  = [[-1,0,1],[-1,0,1],[-1,0,1]]\n"
                "  K135 = [[0,1,1],[-1,0,1],[-1,-1,0]]\n"
                "• Respuesta máxima: max_k |I * K_k|\n"
                "• Brinda tanto magnitud como orientación dominante."
            )

        elif "DoG" in op or "Laplaciano" in op:
            try:
                s1 = float(self.entry_s1.get())
                s2 = float(self.entry_s2.get())
            except ValueError:
                s1, s2 = 0.5, 0.8
            dog, cruces = operador_log_dog(gris, s1, s2)
            self.lbl_tit_b1.config(text="Imagen Original")
            self.lbl_tit_b2.config(text=f"DoG (s1={s1}, s2={s2})")
            self.lbl_tit_b3.config(text="Cruce por Cero (Bordes Finos)")
            self.lbl_tit_b4.config(text="DoG + Cruce por Cero")
            arr_b1 = gris
            arr_b2 = normalizar_u8(dog)
            arr_b3 = cruces
            arr_b4 = np.where(cruces > 0, 255, normalizar_u8(dog) // 2).astype(np.uint8)
            teoria = (
                "LAPLACIANO DE GAUSSIANA (LoG / DoG)\n"
                "• Teoría biológica de David Marr.\n"
                "• Segunda derivada: nabla^2 f = d^2f/dx^2 + d^2f/dy^2\n"
                "• La orilla está exactamente en el CRUCE POR CERO.\n"
                "• DoG aproxima LoG mediante resta de dos Gaussianas:\n"
                "  DoG = G(s1) - G(s2).\n"
                "• Suprime dobles orillas buscando cambios de signo."
            )

        elif "Frei-Chen" in op:
            m_orilla, m_linea, proy = operador_frei_chen_9(gris)
            self.lbl_tit_b1.config(text="W1 (Orilla Horizontal)")
            self.lbl_tit_b2.config(text="W5 (Línea Horizontal)")
            self.lbl_tit_b3.config(text="Subespacio Líneas (W5-W8)")
            self.lbl_tit_b4.config(text="Subespacio Orillas (W1-W4)")
            arr_b1 = normalizar_u8(proy[0])
            arr_b2 = normalizar_u8(proy[4])
            arr_b3 = normalizar_u8(m_linea)
            arr_b4 = normalizar_u8(m_orilla)
            teoria = (
                "MÁSCARAS ORTOGONALES DE FREI-CHEN\n"
                "• Base vectorial de 9 máscaras de 3x3:\n"
                "  - W1 a W4: Subespacio de ORILLAS (escalones).\n"
                "  - W5 a W8: Subespacio de LÍNEAS (grosor 1 px).\n"
                "  - W9: Área UNIFORME (promedio 1/3 ones).\n"
                "• Distingue si una variación es borde o línea\n"
                "  mediante proyección ortogonal R = W^T * Z."
            )

        else:
            # Relajación
            _, _, mag_sobel, _ = operador_sobel_3x3(gris)
            it = int(self.slider_iter_relaj.get())
            relaj = relajacion_bordes_iterativa(mag_sobel, iteraciones=it)
            self.lbl_tit_b1.config(text="Imagen Original")
            self.lbl_tit_b2.config(text="Gradiente Sobel Base")
            self.lbl_tit_b3.config(text=f"Relajación ({it} iteraciones)")
            self.lbl_tit_b4.config(text="Orillas Continuas Reforzadas")
            arr_b1 = gris
            arr_b2 = normalizar_u8(mag_sobel)
            arr_b3 = relaj
            arr_b4 = np.where(relaj > 100, 255, 0).astype(np.uint8)
            teoria = (
                "RELAJACIÓN CONTEXTUAL ITERATIVA\n"
                "• Proceso probabilístico basado en vecindad.\n"
                "• Si un píxel de borde tiene vecinos que también\n"
                "  son bordes alineados, su probabilidad aumenta.\n"
                "• Si es un píxel aislado de ruido, decae a cero.\n"
                "• Conecta contornos rotos y limpia ruido residual."
            )

        self.text_teoria_bordes.insert(tk.END, teoria)

        for arr, lbl in [(arr_b1, self.lbl_img_b1), (arr_b2, self.lbl_img_b2), (arr_b3, self.lbl_img_b3), (arr_b4, self.lbl_img_b4)]:
            im_p = Image.fromarray(arr)
            im_p.thumbnail((360, 220), Image.Resampling.LANCZOS)
            tk_im = ImageTk.PhotoImage(im_p)
            lbl.config(image=tk_im)
            lbl.image = tk_im

    # =========================================================================
    # PESTAÑA 5: DELIMITACIÓN GEOMÉTRICA (USO EXCLUSIVO DE OPENCV)
    # =========================================================================
    def construir_tab5_delimitacion(self):
        panel_ctrl = tk.Frame(self.tab5_geom, bg="#242424", width=340, padx=10, pady=8)
        panel_ctrl.pack(side=tk.LEFT, fill=tk.Y)
        panel_ctrl.pack_propagate(False)

        tk.Label(panel_ctrl, text="Análisis y Delimitación Geométrica", bg="#242424", fg="#ffd700", font=("Arial", 10, "bold")).pack(anchor="w")
        tk.Label(panel_ctrl, text="Uso exclusivo de OpenCV en este módulo:", bg="#242424", fg="#4ec9b0", font=("Arial", 8, "italic")).pack(anchor="w")

        tk.Label(panel_ctrl, text="Método de Segmentación:", bg="#242424", fg="white", font=("Arial", 8, "bold")).pack(anchor="w", pady=(8, 2))
        self.combo_metodo_seg = ttk.Combobox(
            panel_ctrl,
            values=["Otsu Automático (Cualquier Imagen)", "Por Color (Rojo y Amarillo - Caso Sintético)", "Umbral Manual de Intensidad"],
            state="readonly"
        )
        self.combo_metodo_seg.set("Otsu Automático (Cualquier Imagen)")
        self.combo_metodo_seg.pack(fill=tk.X)

        self.slider_umbral_man = tk.Scale(panel_ctrl, from_=0, to=255, orient=tk.HORIZONTAL, bg="#242424", fg="white", highlightthickness=0, troughcolor="#111111")
        self.slider_umbral_man.set(128)
        self.slider_umbral_man.pack(fill=tk.X)

        tk.Button(panel_ctrl, text="⚡ Analizar y Delimitar Regiones", command=self.actualizar_tab5_delimitacion, bg="#0078d4", fg="white", font=("Arial", 9, "bold"), relief=tk.FLAT).pack(fill=tk.X, pady=8)

        tk.Label(panel_ctrl, text="Métricas Extraídas (RegionProps):", bg="#242424", fg="white", font=("Arial", 8, "bold")).pack(anchor="w")
        self.text_props = tk.Text(panel_ctrl, height=9, bg="#111111", fg="#4ec9b0", font=("Consolas", 8))
        self.text_props.pack(fill=tk.X)

        tk.Label(panel_ctrl, text="Código Freeman (8-vecinos):", bg="#242424", fg="#ffd700", font=("Arial", 8, "bold")).pack(anchor="w", pady=(6, 1))
        self.text_freeman = tk.Text(panel_ctrl, height=6, bg="#111111", fg="#dcdcaa", font=("Consolas", 8))
        self.text_freeman.pack(fill=tk.X)

        panel_vis = tk.Frame(self.tab5_geom, bg="#181818", padx=8, pady=8)
        panel_vis.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        panel_vis.columnconfigure(0, weight=2)
        panel_vis.columnconfigure(1, weight=1)
        panel_vis.rowconfigure(0, weight=1)

        f_montaje = tk.Frame(panel_vis, bg="#202020", padx=6, pady=6)
        f_montaje.grid(row=0, column=0, sticky="nsew", padx=4)
        tk.Label(f_montaje, text="Delimitación con Bounding Box, Cuadrado y Contornos", bg="#202020", fg="#4ec9b0", font=("Arial", 10, "bold")).pack()
        self.lbl_img_montaje = tk.Label(f_montaje, bg="#0d0d0d")
        self.lbl_img_montaje.pack(expand=True, fill=tk.BOTH, pady=4)

        f_sub = tk.Frame(panel_vis, bg="#202020", padx=6, pady=6)
        f_sub.grid(row=0, column=1, sticky="nsew", padx=4)
        tk.Label(f_sub, text="Subimagen Binaria 'prop.Image'", bg="#202020", fg="#ffd700", font=("Arial", 10, "bold")).pack()
        self.lbl_img_sub = tk.Label(f_sub, bg="#0d0d0d")
        self.lbl_img_sub.pack(expand=True, fill=tk.BOTH, pady=4)

        self.actualizar_tab5_delimitacion()

    def actualizar_tab5_delimitacion(self):
        r, g, b, gris = separar_canales_rgb(self.img_rgb_original)
        metodo = self.combo_metodo_seg.get()

        if "Otsu" in metodo:
            t_otsu = umbral_otsu_numpy(gris)
            bw_base = (gris < t_otsu).astype(np.uint8) * 255
            bw_solida = imfill_holes(bw_base)
            prop = calcular_propiedades_opencv(bw_solida, etiqueta="Objeto (Otsu)")
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
            self.text_props.insert(tk.END, "No se detectaron regiones conexas mayores a 50 px.")
            self.lbl_img_montaje.config(image="")
            self.lbl_img_sub.config(image="")
            return

        for idx, p in enumerate(lista_props):
            col = colores[idx % len(colores)]
            cnt = p["contorno"]
            x, y, w, h = p["bbox"]
            cx, cy = p["centroide"]

            cv2.drawContours(lienzo, [cnt], -1, col, 2)
            cv2.rectangle(lienzo, (x, y), (x + w, y + h), col, 2)
            xq, yq, lado, _ = calcular_cuadrado_simetrico(p["bbox"], p["centroide"], (alto, ancho))
            cv2.rectangle(lienzo, (xq, yq), (xq + lado, yq + lado), (255, 215, 0), 2)
            cv2.circle(lienzo, (int(round(cx)), int(round(cy))), 5, (255, 0, 0), -1)

            self.text_props.insert(tk.END, f"[{p['etiqueta']}]\n")
            self.text_props.insert(tk.END, f" • Área: {p['area']:.1f} px | Perímetro: {p['perimetro']:.1f}\n")
            self.text_props.insert(tk.END, f" • Centroide: ({cx:.2f}, {cy:.2f})\n")
            self.text_props.insert(tk.END, f" • BBox: [x={x}, y={y}, w={w}, h={h}]\n")
            self.text_props.insert(tk.END, f" • Cuadrado (max w,h): [lado={lado}]\n\n")

            cadena_str = "".join(str(d) for d in p["freeman"][:50])
            self.text_freeman.insert(tk.END, f"{p['etiqueta']} (Primeros 50):\n{cadena_str}...\n\n")

        im_pil = Image.fromarray(lienzo)
        im_pil.thumbnail((500, 480), Image.Resampling.LANCZOS)
        self.tk_montaje = ImageTk.PhotoImage(im_pil)
        self.lbl_img_montaje.config(image=self.tk_montaje)

        sub_im = lista_props[0]["subimagen"]
        sub_pil = Image.fromarray(sub_im)
        sub_pil.thumbnail((250, 250), Image.Resampling.NEAREST)
        self.tk_sub = ImageTk.PhotoImage(sub_pil)
        self.lbl_img_sub.config(image=self.tk_sub)


if __name__ == "__main__":
    raiz = tk.Tk()
    app = AplicacionPDIMaestra(raiz)
    raiz.mainloop()