from ultralytics import YOLO




def freeze_layer(trainer):
    model = trainer.model
    num_freeze = 10 
    freeze = [f'model.{x}.' for x in range(num_freeze)]
    for k, v in model.named_parameters():
        v.requires_grad = True
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False
    print(f"{num_freeze} layers are freezed.")

model = YOLO('yolo11n.pt')
model.add_callback("on_train_start", freeze_layer)
results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=416,
    batch=1,
    device=0,
    patience=10
)
model.export(format='onnx', imgsz=416, half=True)
