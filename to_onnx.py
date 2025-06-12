from ultralytics import YOLO

model = YOLO('best.pt')

#model.export(format='onnx', imgsz=320, half=True)

model.export(
    format="onnx",
    imgsz=544,  # Dimensão fixa baseada no  modelo
    half=True,  # FP16 para performance
    workspace=2,  # 2GB workspace
    batch=1,    # Batch fixo
    verbose=True)
