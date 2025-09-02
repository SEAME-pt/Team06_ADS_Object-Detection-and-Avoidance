import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

def build_trt_engine_gtx1050ti():
    """Engine TensorRT otimizado para GTX 1050 Ti"""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    onnx_path = "best.onnx"
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
    
    # Configuração específica para GTX 1050 Ti
    config = builder.create_builder_config()
    
    # CORREÇÃO: Workspace maior para detecção adequada
    workspace_size = 2 * 1024 * 1024 * 1024  # 2GB para GTX 1050 Ti
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    
    # CORREÇÃO: Configurações de precisão mais conservadoras
    config.set_flag(trt.BuilderFlag.FP16)
    # Não usar INT8 na GTX 1050 Ti - causa problemas de detecção
    
    # CORREÇÃO: Profile com shapes corretos
    profile = builder.create_optimization_profile()
    input_shape = (1, 3, 352, 352)
    
    # Definir shapes com nomes corretos
    input_name = network.get_input(0).name
    profile.set_shape(input_name, input_shape, input_shape, input_shape)
    config.add_optimization_profile(profile)
    
    # CORREÇÃO: Configurações adicionais para estabilidade
    config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    
    print("Construindo engine TensorRT otimizado...")
    print(f"Input name: {input_name}")
    print(f"Input shape: {input_shape}")
    print(f"Workspace: {workspace_size / (1024**3):.1f}GB")
    
    try:
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("Falha ao construir engine TensorRT")
            return None
        
        engine_path = "best_fixed.engine"
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
        
        print(f"✓ Engine TensorRT salvo como: {engine_path}")
        
        # CORREÇÃO: Validar engine imediatamente
        if validate_engine(engine_path):
            return engine_path
        else:
            print("Engine criado mas falhou na validação")
            return None
        
    except Exception as e:
        print(f"Erro durante construção do engine: {e}")
        return None

def validate_engine(engine_path):
    """Validar se o engine foi criado corretamente"""
    try:
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        engine = runtime.deserialize_cuda_engine(engine_data)
        if engine is None:
            print("Engine não pôde ser deserializado")
            return False
        
        context = engine.create_execution_context()
        if context is None:
            print("Contexto não pôde ser criado")
            return False
        
        # Verificar shapes
        print(f"Engine validado:")
        print(f"  - Inputs: {engine.num_bindings}")
        for i in range(engine.num_bindings):
            if engine.binding_is_input(i):
                shape = engine.get_binding_shape(i)
                print(f"  - Input {i}: {shape}")
            else:
                shape = engine.get_binding_shape(i)
                print(f"  - Output {i}: {shape}")
        
        return True
        
    except Exception as e:
        print(f"Erro na validação: {e}")
        return False


def recalibrate_tensorrt_model():
    from ultralytics import YOLO
 
    from ultralytics import YOLO
    import torch

    # Limpar memória primeiro
    torch.cuda.empty_cache()

    model = YOLO("best.pt")

    # Configurações otimizadas para GTX 1050 Ti
    model.export(
        format="engine",
        imgsz=352,
        half=True,          # FP16 essencial para economizar memória
        workspace=1,        # Apenas 1GB workspace para sua GPU
        batch=1,            # Batch mínimo
        device=0,
        verbose=True,
        dynamic=False,      # Shapes fixas
        simplify=True       # Simplificar ONNX
    )


if __name__ == "__main__":
    engine_path = build_trt_engine_gtx1050ti()
    #recalibrate_tensorrt_model()
 

