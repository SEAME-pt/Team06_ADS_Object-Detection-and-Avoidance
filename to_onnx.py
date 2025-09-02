from ultralytics import YOLO

model = YOLO('best.pt')

#model.export(format='onnx', imgsz=320, half=True)

model.export(
    format="onnx",
    imgsz=640,  # Dimensão fixa baseada no  modelo
    half=True,  # FP16 para performance
   # workspace=2,  # 2GB workspace
    batch=1,    # Batch fixo
    verbose=True)


# yolo train data=dataset.yaml model=yolov8n.pt     epochs=500 patience=15 imgsz=352 fliplr=0.0 flipud=0.0 show=True save_period=5

