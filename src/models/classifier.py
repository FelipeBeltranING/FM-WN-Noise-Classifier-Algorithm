import numpy as np

def _normalize(v):
    return v / (np.linalg.norm(v) + 1e-10)

def classify():
    PATH_FM  = "src/dataSet/fmVector.txt"
    PATH_WN  = "src/dataSet/wnVector.txt"
    PATH_MIC = "src/dataSet/micProcessed.txt"

    print("Cargando vectores de referencia...")
    fm_norm  = np.loadtxt(PATH_FM)
    wn_norm  = np.loadtxt(PATH_WN)

    print("Cargando vector del microfono...")
    mic_norm = np.loadtxt(PATH_MIC)

    size     = min(len(mic_norm), len(fm_norm), len(wn_norm))
    mic_norm = mic_norm[:size]
    fm_norm  = fm_norm[:size]
    wn_norm  = wn_norm[:size]

    mic_norm = _normalize(mic_norm)
    fm_norm  = _normalize(fm_norm)
    wn_norm  = _normalize(wn_norm)

    dist_fm = np.mean(np.abs(mic_norm - fm_norm))
    dist_wn = np.mean(np.abs(mic_norm - wn_norm))

    print(f"Distancia con FM: {dist_fm:.5f}")
    print(f"Distancia con Ruido Blanco: {dist_wn:.5f}")

    resultado = "FM" if dist_fm < dist_wn else "Ruido Blanco"
    print(f"Resultado: {resultado}")

    return resultado, dist_fm, dist_wn, mic_norm, fm_norm, wn_norm
