import onnxruntime as ort
import cv2
import numpy as np

def preprocess_image(image, input_size):
    img = cv2.resize(image, input_size)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    return img

# Carregar modelo ONNX
session = ort.InferenceSession("onnx_engine/stop_noEntry.onnx")
input_name = session.get_inputs()[0].name

# Carregar imagem de teste
image = cv2.imread("stop_signal03.jpg")  # Substitua por uma imagem com sinais
input_image = preprocess_image(image, (416, 416))

# Inferência
outputs = session.run(None, {input_name: input_image})[0]
print(outputs.shape, outputs)  # Verifique o formato da saída