import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

def test_engine_performance_fixed(engine_path, input_shape):
    """Teste de performance corrigido para Jetson Nano"""
    
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    
    # Carregar engine
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    
    engine = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()
    

    input_size = int(np.prod(input_shape) * np.dtype(np.float16).itemsize)
    output1_size = int(np.prod((1, 2, input_shape[2], input_shape[3])) * np.dtype(np.float16).itemsize)
    output2_size = output1_size
    
    print(f"Tamanhos de buffer: input={input_size}, output1={output1_size}, output2={output2_size}")
    
 
    try:
        d_input = cuda.mem_alloc(input_size)
        d_output1 = cuda.mem_alloc(output1_size)
        d_output2 = cuda.mem_alloc(output2_size)
    except Exception as e:
        print(f"Erro na alocação de memória: {e}")
        return None
    
    # Bindings
    bindings = [int(d_input), int(d_output1), int(d_output2)]
    
 
    h_input = np.random.randn(*input_shape).astype(np.float16)
    
    try:
        # Copiar para GPU
        cuda.memcpy_htod(d_input, h_input)
        
        # Teste de performance
        import time
        
        # Warm-up
        for _ in range(5):  # Reduzido para Jetson Nano
            context.execute_v2(bindings=bindings)
        
        # Sincronizar antes do benchmark
        cuda.Context.synchronize()
        
        # Benchmark
        start_time = time.time()
        num_iterations = 50  # Reduzido para Jetson Nano
        
        for _ in range(num_iterations):
            context.execute_v2(bindings=bindings)
        
        # Sincronizar após benchmark
        cuda.Context.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations
        fps = 1.0 / avg_time
        
        print(f"✓ Performance do engine {engine_path}:")
        print(f"  - Tempo médio: {avg_time*1000:.2f}ms")
        print(f"  - FPS: {fps:.1f}")
        
        return fps
        
    except Exception as e:
        print(f"Erro durante inferência: {e}")
        return None
    
    finally:
        # CORREÇÃO: Limpeza adequada de recursos
        try:
            if 'd_input' in locals():
                d_input.free()
            if 'd_output1' in locals():
                d_output1.free()
            if 'd_output2' in locals():
                d_output2.free()
        except:
            pass

def build_trt_engine_gtx1050ti():
    """Construção do engine TensorRT para GTX 1050 Ti"""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # Carregar seu modelo ONNX
    onnx_path = "best.onnx"  # Seu modelo
    try:
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print("Erro ao fazer parse do ONNX:")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
    except FileNotFoundError:
        print(f"Arquivo {onnx_path} não encontrado")
        return None
    
    # Configuração para GTX 1050 Ti (4GB VRAM)
    config = builder.create_builder_config()
    workspace_size =512# jeson 512 * 1024 * 1024  # 512MB para GTX 1050 Ti
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    config.set_flag(trt.BuilderFlag.FP16)
    
 
    profile = builder.create_optimization_profile()
    input_shape = (1, 3, 544,544) 
    profile.set_shape("images", input_shape, input_shape, input_shape)
    config.add_optimization_profile(profile)
    
    print("Construindo engine TensorRT para GTX 1050 Ti...")
    
    try:
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("Falha ao construir engine TensorRT")
            return None
        
        engine_path = "best.engine"
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
        
        print(f"✓ Engine TensorRT salvo como: {engine_path}")
        return engine_path
        
    except Exception as e:
        print(f"Erro durante construção do engine: {e}")
        return None

def recalibrate_tensorrt_model():
    from ultralytics import YOLO
    
    # Carregar modelo original
    model = YOLO("best.pt")
    
    model.export(
        format="engine",
        imgsz=544,  # Dimensão fixa baseada no  modelo
        half=True,  # FP16 para performance
        workspace=2,  # 2GB workspace
        batch=1,    # Batch fixo
        verbose=True
    )
    print("Modelo recalibrado com dados específicos")


if __name__ == "__main__":
    #engine_path = build_trt_engine_gtx1050ti()
    recalibrate_tensorrt_model()
 

