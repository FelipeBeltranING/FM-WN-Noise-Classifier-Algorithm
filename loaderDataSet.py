import process

def loaderDataSet():
    
    pathsFM = process.obtainPaths("FM")
    pathsWN = process.obtainPaths("WN")

    print(f"FM: {len(pathsFM)} archivos")
    print(f"WN: {len(pathsWN)} archivos")

    resultsFM = []
    resultsWN = []

    for f in pathsFM:
        process.fillArray(f, resultsFM)

    
    for f in pathsWN:
        process.fillArray(f, resultsWN)

    avg_fm = process.calcAvg(resultsFM)
    avg_wn = process.calcAvg(resultsWN)
    
    print(f"\nNorma promedio FM : {avg_fm['norm']:.5f}")
    print(f"Norma promedio WN : {avg_wn['norm']:.5f}")

    print(f"\nShape acov FM     : {avg_fm['acov'].shape}")
    print(f"Shape fourier FM  : {avg_fm['fourier'].shape}")

    print(f"\nShape acov WN     : {avg_wn['acov'].shape}")
    print(f"Shape fourier WN  : {avg_wn['fourier'].shape}")

loaderDataSet()