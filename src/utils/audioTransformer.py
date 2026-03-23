import os
import numpy as np
import threading
import src.processing.process as process

SR = 44100
DURACION_SEG = 2
FRAMES_VENTANA = SR * DURACION_SEG

BASE = os.path.dirname(os.path.abspath(__file__))
PATH_MIC = os.path.join(BASE, "..", "dataSet", "micProcessed.txt")


class AudioTransformer:
    def __init__(self):
        self._buffer = np.array([], dtype=np.float64)
        self._resultados = []                               
        self._lock = threading.Lock()                 

    def agregar(self, bloque: np.ndarray):
        with self._lock:
            self._buffer = np.concatenate((self._buffer, bloque))
            while len(self._buffer) >= FRAMES_VENTANA:
                ventana      = self._buffer[:FRAMES_VENTANA]
                self._buffer = self._buffer[FRAMES_VENTANA:]
                self._procesarVentana(ventana)

    def detener(self) -> str:
        with self._lock:
            n = len(self._resultados)

        if n == 0:
            raise RuntimeError("No hay fragmentos procesados. El audio fue menor a 2 segundos.")

        with self._lock:
            avg = process.calcAvgVector(self._resultados)

        os.makedirs(os.path.dirname(PATH_MIC), exist_ok=True)
        np.savetxt(PATH_MIC, avg["norm"])
        return PATH_MIC

    def cantidadFragmentos(self) -> int:
        with self._lock:
            return len(self._resultados)

    def reset(self):
        with self._lock:
            self._buffer = np.array([], dtype=np.float64)
            self._resultados = []

    def _procesarVentana(self, ventana: np.ndarray):
        acov = process.calcAutocovariance(ventana)
        fourier = process.calcFourier(acov)
        norm = process.calcNorm(fourier)

        self._resultados.append([acov, fourier, norm])