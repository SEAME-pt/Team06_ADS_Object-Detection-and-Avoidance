import json
import os
import glob

# Diretório das anotações
ANN_DIR = "/home/ndo-vale/Desktop/Team06/datasets/MTSD/mtsd_annotation/mtsd_v2_fully_annotated/annotations"

# Conjunto para armazenar rótulos únicos
labels = set()

# Iterar por todos os arquivos JSON
json_files = glob.glob(f"{ANN_DIR}/*.json")
for json_path in json_files:
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        for obj in data.get("objects", []):
            label = obj.get("label")
            if label:
                labels.add(label)
    except json.JSONDecodeError:
        print(f"Erro ao decodificar: {json_path}")
    except Exception as e:
        print(f"Erro em {json_path}: {e}")

# Exibir rótulos únicos
print("Rótulos encontrados:")
for label in sorted(labels):
    print(f"  {label}")
print(f"Total de rótulos únicos: {len(labels)}")