import torch
from ultralytics import YOLO


model = YOLO("best.pt")  # ou yolov5s.pt

# Shape de entrada (exemplo)
img = torch.zeros((1, 3, 640, 640))
output = model(img)

# Shape de saída
if isinstance(output, (tuple, list)):
    for i, out in enumerate(output):
        print(f"Output {i} shape:", out.shape)
else:
    print("Output shape:", output.shape)

