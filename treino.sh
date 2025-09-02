yolo train data=dataset.yaml model=yolov11x.pt epochs=200 imgsz=640 patience=15 batch=2 box=12.0

yolo train data=dataset.yaml model=yolo11n.pt epochs=200 imgsz=640 patience=15 batch=2

yolo train data=dataset.yaml model=yolo11n.pt epochs=200 imgsz=640 patience=15 batch=2 device=0 workers=4 lr0=0.005

yolo train model=yolov11n.pt data=dataset.yaml epochs=30 imgsz=640 batch=2 patience=8 save_period=5 degrees=10 translate=0.15 scale=0.3 flipud=0.0 fliplr=0.3 mosaic=0.5 mixup=0.05 name=sinais_otimizado
