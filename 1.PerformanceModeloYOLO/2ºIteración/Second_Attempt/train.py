if __name__ == "__main__":
    from ultralytics import YOLO

    # Cargar el modelo 
    model = YOLO("yolov8n.pt")

    # Entrenar el modelo
    model.train(
        data="C:/Users/Ismael Marín/OneDrive/Desktop/src/yolov10/data/config.yaml",
        epochs=50,
        batch=4,
        imgsz=1024,
        device=0,  # Usa la GPU
        workers=3,# Multiprocesador
        optimizer="AdamW",
        lr0=0.001,  # Reducir la tasa de aprendizaj
        amp=False
    )
