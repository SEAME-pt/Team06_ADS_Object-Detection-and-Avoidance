import json
import os
import cv2
import glob
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configurações
DATASET_DIR = "/home/ndo-vale/Desktop/Team06/datasets/TT100k"  # Ajuste para o caminho correto da pasta TT100K
OUTPUT_DIR = "/home/ndo-vale/Desktop/Team06/datasets/TT100k_yolo_stop_ww_ne"  # Diretório de saída
IMG_SIZE = None  # Manter imagens no tamanho original (2048x2048)
TRAIN_SPLIT = 0.8  # Proporção para treino (80% train, 20% valid)

# Classes explícitas
EXPLICIT_CLASSES = ["pne", "p26", "w55"]  # NoEntry, Stop Sign, Pedestrian Crossing

# Função para obter todas as classes de limites de velocidade
""" def get_speed_limit_classes(annotation_files):
    speed_limit_classes = set()
    for ann_file in annotation_files:
        try:
            with open(ann_file, "r") as f:
                data = json.load(f)
            objects = data.get("objects", [])
            if not objects:
                continue
            for obj in objects:
                class_title = obj.get("classTitle")
                if class_title and (class_title.startswith("pl")):
                    speed_limit_classes.add(class_title)
        except Exception as e:
            print(f"Erro ao processar {ann_file}: {e}")
    return sorted(list(speed_limit_classes)) """

# Função para criar nomes legíveis
def get_class_name(class_title):
    if class_title == "pne":
        return "NoEntry"
    elif class_title == "p26":
        return "Stop Sign"
    elif class_title == "w55":
        return "Pedestrian Crossing"
#    elif class_title.startswith("pl"):
 #       return f"Speed Limit {class_title[2:]}"
    return class_title

# Redimensionar imagem e ajustar caixas
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
        x, y, w, h = box
        x = (x * ratio) + pad_w
        y = (y * ratio) + pad_h
        w = w * ratio
        h = h * ratio
        new_boxes.append([x, y, w, h])
    
    return img_padded, new_boxes

# Converter para formato YOLO
def convert_to_yolo_format(boxes, img_width, img_height):
    """Converte bounding boxes para formato YOLO (normalizado)."""
    yolo_boxes = []
    for box in boxes:
        x, y, w, h = box
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        w_norm = w / img_width
        h_norm = h / img_height
        yolo_boxes.append([x_center, y_center, w_norm, h_norm])
    return yolo_boxes

# Processar o dataset
def process_tt100k():
    # Obter classes de limites de velocidade para train
    train_ann_dir = f"{DATASET_DIR}/train/ann"
    train_annotation_files = glob.glob(f"{train_ann_dir}/*.json")
    if not train_annotation_files:
        print(f"Nenhum arquivo de anotações encontrado em: {train_ann_dir}")
        return
    
    #speed_limit_classes = get_speed_limit_classes(train_annotation_files)
    TARGET_CLASSES = EXPLICIT_CLASSES #+ speed_limit_classes
    
    # Criar mapeamento de classes
    CLASS_MAP = {cls: idx for idx, cls in enumerate(TARGET_CLASSES)}
    CLASS_NAMES = [get_class_name(cls) for cls in TARGET_CLASSES]
    
    # Filtrar imagens de treino com classes desejadas
    train_images = []
    for ann_file in train_annotation_files:
        try:
            with open(ann_file, "r") as f:
                img_data = json.load(f)
            objects = img_data.get("objects", [])
            if not objects:
                continue
            has_target = False
            for obj in objects:
                class_title = obj.get("classTitle")
                if class_title in TARGET_CLASSES:
                    has_target = True
                    break
            if has_target:
                img_id = Path(ann_file).stem
                img_data["path"] = f"train/img/{img_id}"
                train_images.append(img_data)
        except Exception as e:
            print(f"Erro ao processar {ann_file}: {e}")
    
    # Dividir em train e valid
    train_images, valid_images = train_test_split(
        train_images, train_size=TRAIN_SPLIT, random_state=42
    )
    
    print(f"Imagens de treino: {len(train_images)}")
    print(f"Imagens de validação: {len(valid_images)}")
    
    # Processar test
    test_ann_dir = f"{DATASET_DIR}/test/ann"
    test_annotation_files = glob.glob(f"{test_ann_dir}/*.json")
    test_images = []
    for ann_file in test_annotation_files:
        try:
            with open(ann_file, "r") as f:
                img_data = json.load(f)
            objects = img_data.get("objects", [])
            if not objects:
                continue
            has_target = False
            for obj in objects:
                class_title = obj.get("classTitle")
                if class_title in TARGET_CLASSES:
                    has_target = True
                    break
            if has_target:
                img_id = Path(ann_file).stem
                img_data["path"] = f"test/img/{img_id}"
                test_images.append(img_data)
        except Exception as e:
            print(f"Erro ao processar {ann_file}: {e}")
    
    print(f"Imagens de teste: {len(test_images)}")
    
    # Processar cada split (train, valid, test)
    for split, images in [("train", train_images), ("valid", valid_images), ("test", test_images)]:
        output_img_dir = Path(OUTPUT_DIR) / split / "images"
        output_label_dir = Path(OUTPUT_DIR) / split / "labels"
        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)
        
        for img_data in images:
            img_id = img_data["path"].split("/")[-1]
            img_path = f"{DATASET_DIR}/{img_data['path']}"
            if not os.path.exists(img_path):
                print(f"Imagem não encontrada: {img_path}")
                continue
            
            # Carregar imagem
            img = cv2.imread(img_path)
            if img is None:
                print(f"Falha ao carregar imagem: {img_path}")
                continue
            img_height = img_data.get("size", {}).get("height", 2048)
            img_width = img_data.get("size", {}).get("width", 2048)
            
            # Extrair bounding boxes
            boxes = []
            class_ids = []
            objects = img_data.get("objects", [])
            for obj in objects:
                class_title = obj.get("classTitle")
                if class_title not in TARGET_CLASSES:
                    continue
                points = obj.get("points", {}).get("exterior", [])
                if len(points) != 2:
                    print(f"Aviso: Bounding box inválido em {img_id}: {points}")
                    continue
                x_min, y_min = points[0]
                x_max, y_max = points[1]
                w = x_max - x_min
                h = y_max - y_min
                x = x_min
                y = y_min
                boxes.append([x, y, w, h])
                class_ids.append(CLASS_MAP[class_title])
            
            if not boxes:
                continue
            
            # Redimensionar imagem e ajustar caixas
            img_processed, boxes = resize_image_and_boxes(img, boxes, IMG_SIZE)
            new_height, new_width = img_processed.shape[:2]
            
            # Converter para formato YOLO
            yolo_boxes = convert_to_yolo_format(boxes, new_width, new_height)
            
            # Salvar imagem
            output_img_path = output_img_dir / img_id
            cv2.imwrite(str(output_img_path), img_processed)
            
            # Salvar anotações
            output_label_path = output_label_dir / f"{img_id.split('.')[0]}.txt"
            with open(output_label_path, "w") as f:
                for class_id, box in zip(class_ids, yolo_boxes):
                    f.write(f"{class_id} {' '.join(map(str, box))}\n")
    
        print(f"Classes incluídas em {split}: {', '.join(CLASS_NAMES)}")
    
    # Criar data.yaml
    data_yaml = {
        "train": str(Path(OUTPUT_DIR) / "train" / "images"),
        "val": str(Path(OUTPUT_DIR) / "valid" / "images"),
        "test": str(Path(OUTPUT_DIR) / "test" / "images"),
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES
    }
    with open(f"{OUTPUT_DIR}/data.yaml", "w") as f:
        json.dump(data_yaml, f, indent=2)

def main():
    process_tt100k()
    print(f"Dataset preparado em: {OUTPUT_DIR}")
    print(f"Use o arquivo {OUTPUT_DIR}/data.yaml para treinar o YOLOv5.")

if __name__ == "__main__":
    main()