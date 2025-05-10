import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Inicializar TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
engine_file_path = "best.engine"

# Carregar o engine
with open(engine_file_path, "rb") as f:
    engine_data = f.read()

runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(engine_data)
context = engine.create_execution_context()

# Descobrir input e output bindings para TensorRT 10
for idx in range(engine.num_io_tensors):
    tensor_name = engine.get_tensor_name(idx)
    tensor_shape = engine.get_tensor_shape(tensor_name)
    tensor_dtype = engine.get_tensor_dtype(tensor_name)
    tensor_mode = engine.get_tensor_mode(tensor_name)

    if tensor_mode == trt.TensorIOMode.INPUT:
        input_name = tensor_name
        input_shape = tensor_shape
    else:
        output_name = tensor_name
        output_shape = tensor_shape

# ---- PRE-PROCESSAR IMAGEM ----
img = cv2.imread('cao_homem.jpg')
img_resized = cv2.resize(img, (640, 640))  # Resize para o esperado
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
img_normalized = img_rgb.astype(np.float32) / 255.0
img_transposed = np.transpose(img_normalized, (2, 0, 1))
img_input = np.expand_dims(img_transposed, axis=0)  # (1, 3, 640, 640)
img_input = np.ascontiguousarray(img_input)

# ---- ALOCAR MEMÓRIA CUDA ----
d_input = cuda.mem_alloc(img_input.nbytes)
d_output = cuda.mem_alloc(np.empty(output_shape, dtype=np.float32).nbytes)

# ---- PREPARAR STREAM e BINDINGS ----
stream = cuda.Stream()
cuda.memcpy_htod_async(d_input, img_input, stream)

context.set_tensor_address(input_name, int(d_input))
context.set_tensor_address(output_name, int(d_output))

context.execute_async_v3(stream.handle)

# ---- COPIAR OUTPUT para CPU ----
output = np.empty(output_shape, dtype=np.float32)
cuda.memcpy_dtoh_async(output, d_output, stream)

# ---- ESPERAR STREAM CONCLUIR ----
stream.synchronize()

# ---- OUTPUT PRONTO ----
print("Output shape:", output.shape)
print("Output sample:", output.flatten()[:10])


