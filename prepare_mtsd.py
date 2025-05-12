import json
import os
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
import shutil
import glob

# Configurações
DATASET_DIR = "/home/ndo-vale/Desktop/Team06/datasets/MTSD"  # Caminho do MTSD
OUTPUT_DIR = "/home/ndo-vale/Desktop/Team06/datasets/MTSD_yolo"  # Diretório de saída
IMG_SIZE = None  # Manter tamanho original (None) ou definir como (416, 416)
TRAIN_SPLIT = 0.8  # Proporção para treino (80% train, 20% valid do train)

# Classes desejadas
RAW_CLASSES = [
    "regulatory--no-entry--g1",
    "warning--pedestrians-crossing--g1",
    "warning--pedestrians-crossing--g4",
    "warning--pedestrians-crossing--g5",
    "warning--pedestrians-crossing--g9",
    "warning--pedestrians-crossing--g10",
    "warning--pedestrians-crossing--g11",
    "warning--pedestrians-crossing--g12",
    "regulatory--stop--g1",
    "regulatory--stop--g2",
    "regulatory--stop--g10"
]
CLASSES = [
    "NoEntry",
    "Pedestrian Crossing",
    "Stop Sign"
]
# Mapear rótulos brutos para índices das classes simplificadas
CLASS_MAP = {
    "regulatory--no-entry--g1": 0,  # NoEntry
    "warning--pedestrians-crossing--g1": 1,  # Pedestrian Crossing
    "warning--pedestrians-crossing--g4": 1,
    "warning--pedestrians-crossing--g5": 1,
    "warning--pedestrians-crossing--g9": 1,
    "warning--pedestrians-crossing--g10": 1,
    "warning--pedestrians-crossing--g11": 1,
    "warning--pedestrians-crossing--g12": 1,
    "regulatory--stop--g1": 2,  # Stop Sign
    "regulatory--stop--g2": 2,
    "regulatory--stop--g10": 2
}
CLASS_NAME_MAP = {
    "regulatory--no-entry--g1": "NoEntry",
    "warning--pedestrians-crossing--g1": "Pedestrian Crossing",
    "warning--pedestrians-crossing--g4": "Pedestrian Crossing",
    "warning--pedestrians-crossing--g5": "Pedestrian Crossing",
    "warning--pedestrians-crossing--g9": "Pedestrian Crossing",
    "warning--pedestrians-crossing--g10": "Pedestrian Crossing",
    "warning--pedestrians-crossing--g11": "Pedestrian Crossing",
    "warning--pedestrians-crossing--g12": "Pedestrian Crossing",
    "regulatory--stop--g1": "Stop Sign",
    "regulatory--stop--g2": "Stop Sign",
    "regulatory--stop--g10": "Stop Sign"
}

# Função para redimensionar imagem e ajustar caixas
def resize_image_and_boxes(img, boxes, target_size):
    """Redimensiona a imagem e ajusta as bounding boxes."""
    h, w = img.shape[:2]
    if target_size is None:
        return img, boxes
    
    # Calcular proporção e padding
    ratio = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Adicionar padding
    pad_w = (target_size[0] - new_w) // 2
    pad_h = (target_size[1] - new_h) // 2
    img_padded = cv2.copyMakeBorder(
        img_resized, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(128, 128, 128)
    )
    
    # Ajustar bounding boxes
    new_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        x_min = (x_min * ratio) + pad_w
        y_min = (y_min * ratio) + pad_h
        x_max = (x_max * ratio) + pad_w
        y_max = (y_max * ratio) + pad_h
        new_boxes.append([x_min, y_min, x_max, y_max])
    
    return img_padded, new_boxes

# Converter para formato YOLO
def convert_to_yolo_format(boxes, img_width, img_height):
    """Converte bounding boxes para formato YOLO (normalizado)."""
    yolo_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        w_norm = (x_max - x_min) / img_width
        h_norm = (y_max - y_min) / img_height
        yolo_boxes.append([x_center, y_center, w_norm, h_norm])
    return yolo_boxes

# Processar o dataset MTSD
def process_mtsd():
    # Diretório das anotações
    ann_dir = f"{DATASET_DIR}/mtsd_annotation/mtsd_v2_fully_annotated/annotations"
    if not os.path.exists(ann_dir):
        print(f"Diretório de anotações não encontrado: {ann_dir}")
        return False
    
    # Diretórios de imagens
    train_img_dirs = [
        f"{DATASET_DIR}/mtsd_train0",
        f"{DATASET_DIR}/mtsd_train1",
        f"{DATASET_DIR}/mtsd_train2"
    ]
    val_img_dir = [f"{DATASET_DIR}/mtsd_images_val"]
    test_img_dir = [f"{DATASET_DIR}/mtsd_test"]
    
    # Função para processar um split
    def process_split(split_name, img_dirs):
        output_img_dir = Path(OUTPUT_DIR) / split_name / "images"
        output_label_dir = Path(OUTPUT_DIR) / split_name / "labels"
        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)
        
        img_count = 0
        total_images = 0
        json_missing = 0
        no_classes = 0
        invalid_images = 0
        found_labels = set()
        
        for img_dir in img_dirs:
            img_paths = glob.glob(f"{img_dir}/images/*.jpg")
            total_images += len(img_paths)
            for img_path in img_paths:
                img_name = Path(img_path).stem
                json_path = f"{ann_dir}/{img_name}.json"
                
                if not os.path.exists(json_path):
                    json_missing += 1
                    print(f"Anotação não encontrada: {json_path}")
                    continue
                
                # Carregar anotação
                try:
                    with open(json_path, "r") as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Erro ao decodificar JSON: {json_path}")
                    continue
                
                # Verificar estrutura do JSON
                if "objects" not in data:
                    print(f"JSON sem chave 'objects': {json_path}")
                    continue
                
                # Carregar imagem
                img = cv2.imread(img_path)
                if img is None:
                    invalid_images += 1
                    print(f"Falha ao carregar imagem: {img_path}")
                    continue
                img_height, img_width = img.shape[:2]
                
                # Filtrar objetos com classes desejadas
                boxes = []
                class_ids = []
                for obj in data.get("objects", []):
                    label = obj.get("label")
                    if label:
                        found_labels.add(label)
                    if label not in RAW_CLASSES:
                        continue
                    bbox = obj.get("bbox")
                    if not bbox or not all(k in bbox for k in ["xmin", "ymin", "xmax", "ymax"]):
                        print(f"Bbox inválido em {json_path}: {bbox}")
                        continue
                    x_min, y_min, x_max, y_max = bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]
                    boxes.append([x_min, y_min, x_max, y_max])
                    class_ids.append(CLASS_MAP[label])
                
                if not boxes:
                    no_classes += 1
                    continue
                
                # Redimensionar imagem e ajustar caixas
                img_processed, boxes = resize_image_and_boxes(img, boxes, IMG_SIZE)
                new_height, new_width = img_processed.shape[:2]
                
                # Converter para formato YOLO
                yolo_boxes = convert_to_yolo_format(boxes, new_height, new_width)
                
                # Salvar imagem
                output_img_path = output_img_dir / f"{img_name}.jpg"
                cv2.imwrite(str(output_img_path), img_processed)
                
                # Salvar anotações
                output_label_path = output_label_dir / f"{img_name}.txt"
                with open(output_label_path, "w") as f:
                    for class_id, box in zip(class_ids, yolo_boxes):
                        f.write(f"{class_id} {' '.join(map(str, box))}\n")
                
                img_count += 1
        
        print(f"Processado {split_name}:")
        print(f"  Total de imagens encontradas: {total_images}")
        print(f"  Imagens processadas: {img_count}")
        print(f"  JSONs ausentes: {json_missing}")
        print(f"  Imagens sem classes desejadas: {no_classes}")
        print(f"  Imagens inválidas: {invalid_images}")
        print(f"  Classes desejadas: {', '.join(CLASSES)}")
        print(f"  Rótulos encontrados: {', '.join(sorted(found_labels))}")
        return img_count
    
    # Processar train e dividir em train/valid
    print("Processando conjunto de treino...")
    train_img_count = process_split("train_temp", train_img_dirs)
    
    # Verificar se há imagens para dividir
    train_img_dir = Path(OUTPUT_DIR) / "train_temp" / "images"
    train_label_dir = Path(OUTPUT_DIR) / "train_temp" / "labels"
    train_img_paths = list(train_img_dir.glob("*.jpg"))
    train_img_ids = [p.stem for p in train_img_paths]
    
    if not train_img_ids:
        print("Erro: Nenhuma imagem processada para o conjunto de treino. Verifique os logs acima.")
        return False
    
    train_ids, valid_ids = train_test_split(
        train_img_ids, train_size=TRAIN_SPLIT, random_state=42
    )
    
    # Mover arquivos para train e valid
    for split, ids in [("train", train_ids), ("valid", valid_ids)]:
        split_img_dir = Path(OUTPUT_DIR) / split / "images"
        split_label_dir = Path(OUTPUT_DIR) / split / "labels"
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_label_dir.mkdir(parents=True, exist_ok=True)
        
        for img_id in ids:
            os.rename(
                train_img_dir / f"{img_id}.jpg",
                split_img_dir / f"{img_id}.jpg"
            )
            os.rename(
                train_label_dir / f"{img_id}.txt",
                split_label_dir / f"{img_id}.txt"
            )
    
    # Remover pasta temporária
    shutil.rmtree(Path(OUTPUT_DIR) / "train_temp")
    
    print(f"Imagens de treino: {len(train_ids)}")
    print(f"Imagens de validação: {len(valid_ids)}")
    
    # Processar test
    print("Processando conjunto de teste...")
    test_img_count = process_split("test", test_img_dir)
    
    # Se test estiver vazio, usar val como test
    if test_img_count == 0:
        print("Conjunto de teste vazio, usando validação como teste...")
        test_img_count = process_split("test", val_img_dir)
    
    print(f"Imagens de teste: {test_img_count}")
    
    # Criar data.yaml apenas se houver imagens processadas
    if train_img_count == 0 and test_img_count == 0:
        print("Erro: Nenhum dado processado. O arquivo data.yaml não será gerado.")
        return False
    
    data_yaml = {
        "train": str(Path(OUTPUT_DIR) / "train" / "images"),
        "val": str(Path(OUTPUT_DIR) / "valid" / "images"),
        "test": str(Path(OUTPUT_DIR) / "test" / "images"),
        "nc": len(CLASSES),
        "names": CLASSES
    }
    with open(f"{OUTPUT_DIR}/data.yaml", "w") as f:
        json.dump(data_yaml, f, indent=2)
    
    return True

def main():
    success = process_mtsd()
    if success:
        print(f"Dataset preparado em: {OUTPUT_DIR}")
        print(f"Use o arquivo {OUTPUT_DIR}/data.yaml para treinar o YOLOv8.")
    else:
        print("Falha ao preparar o dataset. Verifique os logs para detalhes.")

if __name__ == "__main__":
    main()