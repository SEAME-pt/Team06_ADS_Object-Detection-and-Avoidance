import cv2

# Carrega o modelo otimizado em TensorRT (.engine)
net = cv2.dnn.readNet("best.engine")  # ou colocar o caminho completo

# Usa TensorRT como backend
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Inicia a câmara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)

    outputs = net.forward()

    # Aqui podes tratar os outputs como quiseres, ou apenas mostrar o frame original
    cv2.imshow("TensorRT YOLOv8 Inference", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
