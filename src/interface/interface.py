import queue
import numpy as np
import sounddevice as sd
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.utils.audioTransformer import AudioTransformer
from src.models.classifier import classify

SR = 44100
BLOCK_SIZE = 1024
FRAMES_2S = SR * 2   # fragmentos necesarios para clasificar

audio_queue = queue.Queue()
transformer = AudioTransformer()


def callback(indata, frames, time, status):
    bloque = indata[:, 0].astype(np.float64)
    audio_queue.put(bloque)
    transformer.agregar(bloque)

class App:

    def __init__(self, root):
        self.root = root
        self.root.title("Reconocedor de audio")
        self.root.configure(bg="#111111")
        self.root.resizable(True, True)

        self.stream = None
        self.ani = None
        self.fig = None
        self.canvas = None

        self._construirLayout()
        self._construirGrafica()

    # ── Layout principal (izquierda controles, derecha gráfica) ───────────────
    def _construirLayout(self):
        # Panel izquierdo
        self.panel_izq = tk.Frame(self.root, bg="#111111", padx=20, pady=20)
        self.panel_izq.pack(side="left", fill="y")

        # Panel derecho
        self.panel_der = tk.Frame(self.root, bg="#1a1a1a")
        self.panel_der.pack(side="right", fill="both", expand=True)

        # ── Botones ───────────────────────────────────────────────────────────
        self.btn_iniciar = tk.Button(
            self.panel_izq,
            text="Empezar a grabar",
            command=self._iniciarGrabacion,
            bg="#1e1e1e", fg="#ffffff",
            activebackground="#2a2a2a", activeforeground="#ffffff",
            relief="flat", padx=12, pady=8,
            font=("Consolas", 10), cursor="hand2",
            width=18
        )
        self.btn_iniciar.pack(anchor="w", pady=(0, 8))

        self.btn_detener = tk.Button(
            self.panel_izq,
            text="Detener",
            command=self._detenerGrabacion,
            bg="#1e1e1e", fg="#555555",
            activebackground="#2a2a2a", activeforeground="#ffffff",
            relief="flat", padx=12, pady=8,
            font=("Consolas", 10), cursor="hand2",
            width=18, state="disabled"
        )
        self.btn_detener.pack(anchor="w", pady=(0, 24))

        # ── Estado de grabación ───────────────────────────────────────────────
        tk.Label(self.panel_izq, text="Estado de grabación",
                 bg="#111111", fg="#444444",
                 font=("Consolas", 8)).pack(anchor="w")

        self.lbl_grabacion = tk.Label(
            self.panel_izq, text="Detenido",
            bg="#111111", fg="#555555",
            font=("Consolas", 10), wraplength=180, justify="left"
        )
        self.lbl_grabacion.pack(anchor="w", pady=(2, 20))

        # ── Estado de suficiencia ─────────────────────────────────────────────
        tk.Label(self.panel_izq, text="Suficiencia para comparación",
                 bg="#111111", fg="#444444",
                 font=("Consolas", 8)).pack(anchor="w")

        self.lbl_suficiencia = tk.Label(
            self.panel_izq, text="—",
            bg="#111111", fg="#555555",
            font=("Consolas", 10), wraplength=180, justify="left"
        )
        self.lbl_suficiencia.pack(anchor="w", pady=(2, 20))

        # ── Resultado ─────────────────────────────────────────────────────────
        tk.Label(self.panel_izq, text="Clasificación",
                 bg="#111111", fg="#444444",
                 font=("Consolas", 8)).pack(anchor="w")

        self.lbl_resultado = tk.Label(
            self.panel_izq, text="—",
            bg="#111111", fg="#555555",
            font=("Consolas", 12, "bold"), wraplength=180, justify="left"
        )
        self.lbl_resultado.pack(anchor="w", pady=(2, 0))

    # ── Gráfica inicial vacía ─────────────────────────────────────────────────
    def _construirGrafica(self):
        self.fig, self.ax = plt.subplots(figsize=(7, 4))
        self._estiloAx(self.ax)
        self.ax.set_title("Esperando grabación...",
                          color="#444444", fontsize=10)
        self.fig.tight_layout()

        self._montarCanvas()

    def _montarCanvas(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.panel_der)
        self.canvas.get_tk_widget().pack(fill="both", expand=True,
                                         padx=12, pady=12)
        self.canvas.draw()

    def _estiloAx(self, ax):
        self.fig.patch.set_facecolor("#1a1a1a")
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="#444444", labelsize=8)
        ax.spines[:].set_color("#2a2a2a")
        ax.grid(True, color="#222222", linewidth=0.5)
        ax.set_xlabel("Frecuencia (Hz)", color="#555555", fontsize=9)
        ax.set_ylabel("Magnitud normalizada", color="#555555", fontsize=9)

    # ── Grabación ─────────────────────────────────────────────────────────────
    def _iniciarGrabacion(self):
        transformer.reset()

        self.btn_iniciar.config(state="disabled")
        self.btn_detener.config(state="normal", fg="#ffffff")
        self.lbl_grabacion.config(text="🔴  Grabando...", fg="#e74c3c")
        self.lbl_suficiencia.config(text="Esperando 2 seg...", fg="#888888")
        self.lbl_resultado.config(text="—", fg="#555555")

        # Línea del espectro acumulado (se actualiza cada 2 seg)
        self.ax.cla()
        self._estiloAx(self.ax)
        self.ax.set_title("Espectro acumulado (mic)", color="#888888", fontsize=10)
        self.line_mic, = self.ax.plot([], [], color="#ff6f61",
                                      linewidth=1.0, label="Micrófono")
        self.ax.set_xlim(0, 8000)
        self.ax.set_ylim(0, 1.1)
        self.canvas.draw()

        self.stream = sd.InputStream(
            samplerate=SR, blocksize=BLOCK_SIZE,
            channels=1, dtype="float32", callback=callback
        )
        self.stream.start()

        # Polling cada 500ms para detectar nuevo fragmento listo
        self._ultimo_fragmento = 0
        self._pollFragmento()

    def _pollFragmento(self):
        """Cada 500 ms revisa si audioTransformer procesó un nuevo fragmento."""
        if self.stream is None:
            return

        n = transformer.cantidadFragmentos()

        # Suficiencia
        if n >= 1:
            self.lbl_suficiencia.config(
                text=f"✅  Listo ({n} fragmento{'s' if n > 1 else ''} de 2 seg)",
                fg="#2ecc71"
            )
        else:
            self.lbl_suficiencia.config(
                text="Esperando 2 seg...", fg="#888888"
            )

        # Actualiza gráfica solo cuando hay un fragmento nuevo
        if n > self._ultimo_fragmento:
            self._ultimo_fragmento = n
            self._actualizarEspectroAcumulado()

        self.root.after(500, self._pollFragmento)

    def _actualizarEspectroAcumulado(self):
        """Recalcula el promedio actual y actualiza la línea del micrófono."""
        avg  = transformer.promedioActual()
        norm = avg["norm"]

        size  = len(norm) // 2
        freqs = np.fft.fftfreq(len(norm), d=1 / SR)[:size]
        norm_half = norm[:size]

        # Normaliza para mantener eje Y entre 0 y 1
        max_val = np.max(norm_half) + 1e-10
        self.line_mic.set_data(freqs, norm_half / max_val)
        self.canvas.draw_idle()

    def _detenerGrabacion(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        self.btn_detener.config(state="disabled")
        self.lbl_grabacion.config(text="⚙  Clasificando...", fg="#f39c12")
        self.root.update()

        try:
            transformer.detener()
            resultado, dist_fm, dist_wn, mic_n, fm_n, wn_n = classify()

            color = "#4fc3f7" if resultado == "FM" else "#aed581"
            self.lbl_grabacion.config(text="Detenido", fg="#555555")
            self.lbl_resultado.config(text=resultado, fg=color)

            self._mostrarComparativa(resultado, dist_fm, dist_wn,
                                     mic_n, fm_n, wn_n)
        except RuntimeError as e:
            self.lbl_grabacion.config(text=f"{e}", fg="#e74c3c")
        finally:
            self.btn_iniciar.config(state="normal")

    # ── Gráfica comparativa final ─────────────────────────────────────────────
    def _mostrarComparativa(self, resultado, dist_fm, dist_wn,
                             mic_n, fm_n, wn_n):
        plt.close(self.fig)
        self.fig, self.ax = plt.subplots(figsize=(7, 4))
        self._estiloAx(self.ax)

        size  = min(len(mic_n), len(fm_n), len(wn_n))
        freqs = np.linspace(0, SR / 2, size)

        alpha_fm = 0.9 if resultado == "FM" else 0.3
        alpha_wn = 0.9 if resultado == "Ruido Blanco" else 0.3

        self.ax.plot(freqs, fm_n[:size], color="#4fc3f7", linewidth=1.1,
                     alpha=alpha_fm, label=f"FM  ({dist_fm:.5f})")
        self.ax.plot(freqs, wn_n[:size], color="#aed581", linewidth=1.1,
                     alpha=alpha_wn, label=f"Ruido Blanco  ({dist_wn:.5f})")
        self.ax.plot(freqs, mic_n[:size], color="#ff6f61", linewidth=1.0,
                     alpha=0.9, label="Micrófono")

        color_titulo = "#4fc3f7" if resultado == "FM" else "#aed581"
        self.ax.set_title(f"Resultado: {resultado}",
                          color=color_titulo, fontsize=11, fontweight="bold")
        self.ax.legend(facecolor="#1a1a1a", edgecolor="#333333",
                       labelcolor="#cccccc", fontsize=8)
        self.fig.tight_layout()
        self._montarCanvas()

    
def launch():
    root = tk.Tk()
    App(root)
    root.protocol("WM_DELETE_WINDOW", lambda: root.quit())
    root.mainloop()
    root.destroy()