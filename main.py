from src.loaders.loaderDataSet import train
from src.interface.interface import launch
from src.models.classifier import classify


if __name__ == "__main__":
    print("1. Entrenar")
    print("2. Grabar audio")

    opcion = input("Selecciona una opcion [1/2]: ").strip()

    if opcion == "1":
        train()
    elif opcion == "2":
        print("Iniciando grabación...")
        launch()         # bloquea hasta que se cierra la ventana
        print("Ventana cerrada, procediendo a clasificar...")
        classify()       # la ventana se cerro, ahora clasificamos
    else:
        print("Opcion invalida")