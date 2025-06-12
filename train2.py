from ultralytics import YOLO

model = YOLO('yolo11n.pt')  
 

# Configurações de augmentation
augment_config = {
    'hsv_h': 0.015,      # Variação de matiz
    'hsv_s': 0.7,        # Variação de saturação
    'hsv_v': 0.4,        # Variação de brilho
    'degrees': 10,       # Rotação máxima
    'translate': 0.1,    # Translação
    'scale': 0.2,        # Escala
    'shear': 2.0,        # Cisalhamento
    'perspective': 0.0,  # Perspectiva
    'flipud': 0.0,       # Flip vertical (não recomendado para sinais)
    'fliplr': 0.0,       # Flip horizontal (cuidado com texto)
    'mosaic': 1.0,       # Mosaic augmentation
    'mixup': 0.1,        # MixUp
    'copy_paste': 0.1    # Copy-paste augmentation
}

training_config = {
    'epochs': 200,
    'patience': 30,      # Early stopping
    'batch': 16,         # Ajustar conforme GPU
    'imgsz': 640,        # Tamanho da imagem
    'save_period': 10,   # Salvar checkpoint a cada 10 epochs
    'cache': True,       # Cache imagens na RAM
    'device': 0,         # GPU 0
    'workers': 4,        # Threads para carregamento
    'project': 'traffic_signs',
    'name': 'yolov8_experiment',
    'exist_ok': True
}



model = YOLO('yolov8s.pt')  # ou yolov8s.pt para melhor precisão

results = model.train(
    data='dataset.yaml',
    epochs=200,
    imgsz=448,
    batch=16,
    patience=30,
    save_period=10,
    cache=True,
    device=0,
    workers=4,

    
    # Augmentations
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10,
    translate=0.1,
    scale=0.2,
    shear=2.0,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    
    # Learning rate e optimizer
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    
    # Loss weights
    box=7.5,
    cls=0.5,
    dfl=1.5,
    
    # Validação
    val=True,
    plots=True,
    save_json=True
)

 

