import sys, types, importlib

def mock_trt_pycuda():
    sys.modules.setdefault('tensorrt', types.SimpleNamespace(Logger=object))
    sys.modules.setdefault('pycuda', types.SimpleNamespace())
    sys.modules.setdefault('pycuda.autoinit', types.SimpleNamespace())
    sys.modules.setdefault('pycuda.driver', types.SimpleNamespace())
    sys.modules.setdefault('pycuda.gpuarray', types.SimpleNamespace())
    sys.modules.setdefault('pycuda.compiler', types.SimpleNamespace())

def test_import_yolo_runtimes():
    mock_trt_pycuda()
    # importa os módulos principais sem precisar de TensorRT
    importlib.import_module('yolo11')
    importlib.import_module('testModel')