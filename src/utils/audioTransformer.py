import os
import wave
import struct

CARPETA_ENTRADA  = "InformationSignals0"        
CARPETA_SALIDA   = "InformationSignals" 
DURACION_SEG     = 2                 
PREFIJO          = "IS"             


def cortar_wav(ruta_entrada, ruta_salida, duracion_seg):
    with wave.open(ruta_entrada, "rb") as wav_in:
        framerate   = wav_in.getframerate()
        n_channels  = wav_in.getnchannels()
        sampwidth   = wav_in.getsampwidth()
        total_frames = wav_in.getnframes()

        frames_necesarios = int(framerate * duracion_seg)

        if total_frames < frames_necesarios:
            wav_in.rewind()
            datos = wav_in.readframes(total_frames)
            silencio_frames = frames_necesarios - total_frames
            silencio = b'\x00' * silencio_frames * n_channels * sampwidth
            datos += silencio
        else:
            wav_in.rewind()
            datos = wav_in.readframes(frames_necesarios)

        with wave.open(ruta_salida, "wb") as wav_out:
            wav_out.setnchannels(n_channels)
            wav_out.setsampwidth(sampwidth)
            wav_out.setframerate(framerate)
            wav_out.writeframes(datos)


def main():
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    archivos = sorted([
        f for f in os.listdir(CARPETA_ENTRADA)
        if f.lower().endswith(".wav")
    ])

    if not archivos:
        print(f"⚠  No se encontraron archivos .wav en '{CARPETA_ENTRADA}'")
        return

    print(f"✔  {len(archivos)} archivo(s) encontrado(s). Procesando...\n")

    for i, archivo in enumerate(archivos, start=1):
        ruta_entrada = os.path.join(CARPETA_ENTRADA, archivo)
        nuevo_nombre = f"{PREFIJO}{i:02d}.wav"         
        ruta_salida  = os.path.join(CARPETA_SALIDA, nuevo_nombre)

        cortar_wav(ruta_entrada, ruta_salida, DURACION_SEG)
        print(f"  {archivo:30s}  →  {nuevo_nombre}")

    print(f"\n✅ Listo. Archivos guardados en '{CARPETA_SALIDA}/'")


if __name__ == "__main__":
    main()