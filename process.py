import os
from statsmodels.tsa.stattools import acovf
import numpy as np
import librosa


def obtainPaths(path):
    return [os.path.join(path, file) for file in os.listdir(path)]

def loadAudio(file_path, sr=44100):
    y, _ = librosa.load(file_path,sr=sr, dtype=np.float64)
    return y

def calcAutocovariance(y):
    return acovf(y,fft=True,demean=True).astype(np.float64)
    
def calcFourier(y):
    return np.abs(np.fft.fft(y)).astype(np.float64)
    
def calcNorm(y):
    return np.float64(np.linalg.norm(y))

def fillArray(f ,results):
    
    y = loadAudio(f)

    results.append([
        calcAutocovariance(y),
        calcFourier(y),
        calcNorm(y)])

def calcAvg(results):

    acovs, fouriers, norms = zip(*results)
    
    return {
        "acov": np.mean(acovs, axis=0),
        "fourier": np.mean(fouriers, axis=0),
        "norm": np.mean(norms)
    }
    

