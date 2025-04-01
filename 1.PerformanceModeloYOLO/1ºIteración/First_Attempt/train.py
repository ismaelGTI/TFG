if __name__ == "__main__":
    from ultralytics import YOLOv10

    # Cargar el modelo 
    model = YOLOv10.from_pretrained('jameslahm/yolov10s')

    # Entrenar el modelo
    model.train(
        data="C:/Users/Ismael Marín/OneDrive/Desktop/src/yolov10/data/config.yaml",
        epochs=50,
        batch=2,
        imgsz=640,
        device=0,  # Usa la GPU
        workers=4,# Multiprocesador
        optimizer="AdamW",
        lr0=0.0001,  # Reducir la tasa de aprendizaj
        amp=False
    )
