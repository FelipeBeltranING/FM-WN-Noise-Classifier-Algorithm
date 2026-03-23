import os
import queue
import numpy as np
import sounddevice as sd
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.utils.audioTransformer import AudioTransformer

SR         = 44100
BLOCK_SIZE = 1024

audio_queue  = queue.Queue()
transformer  = AudioTransformer()

def callback(indata, frames, time, status):
    bloque = indata[:, 0].astype(np.float64)
    audio_queue.put(bloque)
    transformer.agregar(bloque)

# Ventana principal
class App:

    def __init__(self, root):
        self.root = root
        self.root.title("Reconocedor de audio")
        self.root.configure(bg="#111111")
        self.root.resizable(False, False)

        self._construirUI()
        self._construirGrafica()

        self.stream     = None
        self.ani        = None
        self.datos_plot = np.zeros(BLOCK_SIZE)

    def _construirUI(self):
        frame_top = tk.Frame(self.root, bg="#111111", pady=12, padx=16)
        frame_top.pack(fill="x")

        self.btn_iniciar = tk.Button(
            frame_top,
            text="⏺  Iniciar grabación",
            command=self._iniciarGrabacion,
            bg="#c0392b", fg="#ffffff",
            activebackground="#e74c3c", activeforeground="#ffffff",
            relief="flat", padx=14, pady=6,
            font=("Consolas", 11, "bold"), cursor="hand2"
        )
        self.btn_iniciar.pack(side="left")

        self.btn_detener = tk.Button(
            frame_top,
            text="⏹  Detener",
            command=self._detenerGrabacion,
            bg="#333333", fg="#888888",
            activebackground="#444444", activeforeground="#ffffff",
            relief="flat", padx=14, pady=6,
            font=("Consolas", 11, "bold"), cursor="hand2",
            state="disabled"
        )
        self.btn_detener.pack(side="left", padx=(10, 0))

        self.lbl_estado = tk.Label(
            frame_top, text="", bg="#111111",
            fg="#888888", font=("Consolas", 10)
        )
        self.lbl_estado.pack(side="left", padx=16)

        self.lbl_fragmentos = tk.Label(
            frame_top, text="", bg="#111111",
            fg="#555555", font=("Consolas", 10)
        )
        self.lbl_fragmentos.pack(side="right", padx=16)

    def _construirGrafica(self):
        self.fig, self.ax = plt.subplots(figsize=(9, 3))
        self.fig.patch.set_facecolor("#111111")
        self.ax.set_facecolor("#111111")

        self.line_onda, = self.ax.plot(
            np.linspace(0, 1, BLOCK_SIZE),
            np.zeros(BLOCK_SIZE),
            color="#ff6f61", linewidth=0.8
        )

        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlabel("Tiempo (bloque actual)", color="#555555", fontsize=9)
        self.ax.set_ylabel("Amplitud", color="#555555", fontsize=9)
        self.ax.tick_params(colors="#444444", labelsize=8)
        self.ax.spines[:].set_color("#222222")
        self.ax.grid(True, color="#1e1e1e", linewidth=0.5)
        self.fig.tight_layout()

        canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        canvas.get_tk_widget().pack(fill="both", padx=16, pady=(0, 16))
        self.canvas = canvas

    # Grabación
    def _iniciarGrabacion(self):
        transformer.reset()

        self.btn_iniciar.config(state="disabled")
        self.btn_detener.config(state="normal", fg="#ffffff")
        self.lbl_estado.config(text="🔴  Grabando...", fg="#e74c3c")
        self.lbl_fragmentos.config(text="fragmentos: 0")

        self.ani = animation.FuncAnimation(
            self.fig,
            self._actualizarOnda,
            interval=40,
            blit=True,
            cache_frame_data=False
        )
        self.canvas.draw()

        self.stream = sd.InputStream(
            samplerate=SR,
            blocksize=BLOCK_SIZE,
            channels=1,
            dtype="float32",
            callback=callback
        )
        self.stream.start()

        self._actualizarContador()

    def _actualizarContador(self):
        if self.stream is not None:
            n = transformer.cantidadFragmentos()
            self.lbl_fragmentos.config(text=f"fragmentos procesados: {n}")
            self.root.after(1000, self._actualizarContador)

    def _detenerGrabacion(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if self.ani:
            self.ani.event_source.stop()
            self.ani = None

        self.btn_detener.config(state="disabled")
        self.lbl_estado.config(text="⚙  Guardando...", fg="#f39c12")
        self.root.update()

        try:
            ruta = transformer.detener()
            nombre = os.path.basename(ruta)
            self.lbl_estado.config(
                text=f"✅  Guardado en {nombre}", fg="#2ecc71"
            )
        except RuntimeError as e:
            self.lbl_estado.config(text=f"❌  {e}", fg="#e74c3c")
        finally:
            self.btn_iniciar.config(state="normal")

    def _actualizarOnda(self, frame):
        while not audio_queue.empty():
            self.datos_plot = audio_queue.get_nowait()

        self.line_onda.set_ydata(
            self.datos_plot
            if len(self.datos_plot) == BLOCK_SIZE
            else np.zeros(BLOCK_SIZE)
        )
        return (self.line_onda,)

def launch():
    root = tk.Tk()
    App(root)
    root.mainloop()