from ultralytics import YOLO

model = YOLO('yolo11n.pt')  

model.train(
    data='dataset.yaml',  # Arquivo de configuração do dataset
    epochs=100,           
    imgsz=448,            # Tamanho de imagem pequeno para Jetson Nano
    batch=2,              # Batch size pequeno para estabilidade no Nano
    device=0,             # GPU (se disponível no treinamento)
    project='runs/train', # Diretório de saída
    name='exp',           # Nome do experimento
    exist_ok=True,        # Sobrescrever resultados existentes
    optimizer='AdamW',    # Otimizador recomendado para datasets pequenos
    lr0=0.001,            # Taxa de aprendizado inicial
    lrf=0.01,             # Fator final da taxa de aprendizado
    weight_decay=0.001,   # Regularização forte para dataset pequeno
    patience=5,          # Early stopping após 20 épocas sem melhoria
    box=12.0,              # Peso maior para Box Loss (caixas mais precisas)
    #cls=0.3,              # Peso menor para Classification Loss (9 classes)
    cls=0.7,              # Ajuste para classes desbalanceadas
    dfl=1.5,              # Peso padrão para DFL Loss
    augment=True,         # Ativar data augmentation
    mosaic=0.5,           # Probabilidade de usar mosaic augmentation
    flipud=0.5,           # Probabilidade de flip vertical
    fliplr=0.5,           # Probabilidade de flip horizontal
    scale=0.3,      # Simula objetos menores (mais distantes)
    save_period=10,       # Salvar checkpoints a cada 10 épocas
    verbose=False          # Logs detalhados
)

