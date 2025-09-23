import os
import shutil
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import yaml

class YOLODatasetSplitter:
    def __init__(self, source_path, output_path, train_ratio=0.8, val_ratio=0.2, random_seed=42):
        """
        Inicializa o divisor de dataset YOLO
        
        Args:
            source_path: Caminho do dataset original
            output_path: Caminho onde será criado o dataset dividido
            train_ratio: Proporção para treinamento (0.8 = 80%)
            val_ratio: Proporção para validação (0.2 = 20%)
            random_seed: Seed para reprodutibilidade
        """
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.random_seed = random_seed
        
        # Definir classes do dataset
        self.class_names = [
            "STOP", "PASSAGEM", "VEL_50", "VEL_80", "SEMAFORO_VERMELHO", 
            "SEMAFORO_VERDE", "SEMAFORO_LARANJA", "PASSADEIRA", "DANGER", "CURVA"
        ]
        
        # Definir extensões de imagem suportadas
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Estatísticas
        self.stats = {
            'total_images': 0,
            'train_images': 0,
            'val_images': 0,
            'train_labels': defaultdict(int),
            'val_labels': defaultdict(int),
            'missing_labels': [],
            'missing_images': []
        }
        
        random.seed(self.random_seed)
    
    def validate_ratios(self):
        """Valida se as proporções somam 1.0"""
        total = self.train_ratio + self.val_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"As proporções devem somar 1.0. Atual: {total}")
    
    def find_image_label_pairs(self):
        """Encontra todos os pares imagem-label válidos"""
        print("🔍 Procurando pares imagem-label...")
        
        image_files = []
        label_files = []
        
        # Encontrar todas as imagens
        for ext in self.image_extensions:
            image_files.extend(self.source_path.glob(f"**/*{ext}"))
            image_files.extend(self.source_path.glob(f"**/*{ext.upper()}"))
        
        # Encontrar todos os labels
        label_files = list(self.source_path.glob("**/*.txt"))
        
        print(f"📸 Encontradas {len(image_files)} imagens")
        print(f"🏷️  Encontrados {len(label_files)} arquivos de label")
        
        # Criar mapeamento nome -> caminho
        image_map = {f.stem: f for f in image_files}
        label_map = {f.stem: f for f in label_files}
        
        # Encontrar pares válidos
        valid_pairs = []
        
        for name in image_map.keys():
            if name in label_map:
                valid_pairs.append((image_map[name], label_map[name]))
            else:
                self.stats['missing_labels'].append(str(image_map[name]))
        
        for name in label_map.keys():
            if name not in image_map:
                self.stats['missing_images'].append(str(label_map[name]))
        
        print(f"✅ Encontrados {len(valid_pairs)} pares válidos")
        
        if self.stats['missing_labels']:
            print(f"⚠️  {len(self.stats['missing_labels'])} imagens sem labels")
        
        if self.stats['missing_images']:
            print(f"⚠️  {len(self.stats['missing_images'])} labels sem imagens")
        
        return valid_pairs
    
    def analyze_labels(self, label_file):
        """Analisa um arquivo de label e retorna as classes presentes"""
        classes_in_file = set()
        
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        class_id = int(line.split()[0])
                        if 0 <= class_id < len(self.class_names):
                            classes_in_file.add(class_id)
        except Exception as e:
            print(f"⚠️  Erro ao ler {label_file}: {e}")
        
        return classes_in_file
    
    def stratified_split(self, pairs):
        """Realiza divisão estratificada baseada nas classes presentes"""
        print("🎯 Realizando divisão estratificada...")
        
        # Analisar classes em cada arquivo
        file_classes = {}
        class_files = defaultdict(list)
        
        for img_path, label_path in pairs:
            classes = self.analyze_labels(label_path)
            file_classes[img_path.stem] = classes
            
            # Adicionar arquivo a cada classe que contém
            for class_id in classes:
                class_files[class_id].append((img_path, label_path))
        
        # Dividir cada classe proporcionalmente
        train_pairs = set()
        val_pairs = set()
        
        for class_id, class_pairs in class_files.items():
            random.shuffle(class_pairs)
            
            train_count = int(len(class_pairs) * self.train_ratio)
            
            for i, pair in enumerate(class_pairs):
                if i < train_count:
                    train_pairs.add(pair)
                else:
                    val_pairs.add(pair)
        
        # Converter sets para listas
        train_list = list(train_pairs)
        val_list = list(val_pairs)
        
        # Adicionar arquivos que não foram categorizados (sem labels válidos)
        remaining_pairs = set(pairs) - train_pairs - val_pairs
        remaining_list = list(remaining_pairs)
        random.shuffle(remaining_list)
        
        train_remaining = int(len(remaining_list) * self.train_ratio)
        train_list.extend(remaining_list[:train_remaining])
        val_list.extend(remaining_list[train_remaining:])
        
        return train_list, val_list
    
    def create_directory_structure(self):
        """Cria a estrutura de diretórios YOLO"""
        print("📁 Criando estrutura de diretórios...")
        
        directories = [
            self.output_path / "images" / "train",
            self.output_path / "images" / "val",
            self.output_path / "labels" / "train",
            self.output_path / "labels" / "val"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ {directory}")
    
    def copy_files(self, pairs, subset_name):
        """Copia arquivos para o subset especificado (train/val)"""
        print(f"📋 Copiando arquivos para {subset_name}...")
        
        images_dir = self.output_path / "images" / subset_name
        labels_dir = self.output_path / "labels" / subset_name
        
        copied_count = 0
        
        for img_path, label_path in pairs:
            try:
                # Copiar imagem
                img_dest = images_dir / img_path.name
                shutil.copy2(img_path, img_dest)
                
                # Copiar label
                label_dest = labels_dir / label_path.name
                shutil.copy2(label_path, label_dest)
                
                # Atualizar estatísticas
                classes = self.analyze_labels(label_path)
                for class_id in classes:
                    if subset_name == "train":
                        self.stats['train_labels'][class_id] += 1
                    else:
                        self.stats['val_labels'][class_id] += 1
                
                copied_count += 1
                
            except Exception as e:
                print(f"⚠️  Erro ao copiar {img_path.name}: {e}")
        
        print(f"   ✅ {copied_count} pares copiados para {subset_name}")
        
        if subset_name == "train":
            self.stats['train_images'] = copied_count
        else:
            self.stats['val_images'] = copied_count
    
    def create_yaml_config(self):
        """Cria arquivo de configuração YAML para YOLO"""
        print("📝 Criando arquivo de configuração YAML...")
        
        config = {
            'path': str(self.output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.class_names),
            'names': {i: name for i, name in enumerate(self.class_names)}
        }
        
        yaml_path = self.output_path / "dataset.yaml"
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"   ✅ Configuração salva em: {yaml_path}")
        
        # Criar também um arquivo classes.txt
        classes_path = self.output_path / "classes.txt"
        with open(classes_path, 'w') as f:
            for name in self.class_names:
                f.write(f"{name}\n")
        
        print(f"   ✅ Classes salvas em: {classes_path}")
    
    def print_statistics(self):
        """Exibe estatísticas da divisão"""
        print("\n" + "="*60)
        print("📊 ESTATÍSTICAS DA DIVISÃO")
        print("="*60)
        
        print(f"📁 Total de imagens processadas: {self.stats['total_images']}")
        print(f"🏋️  Imagens de treinamento: {self.stats['train_images']} ({self.stats['train_images']/self.stats['total_images']*100:.1f}%)")
        print(f"🧪 Imagens de validação: {self.stats['val_images']} ({self.stats['val_images']/self.stats['total_images']*100:.1f}%)")
        
        if self.stats['missing_labels']:
            print(f"⚠️  Imagens sem labels: {len(self.stats['missing_labels'])}")
        
        if self.stats['missing_images']:
            print(f"⚠️  Labels sem imagens: {len(self.stats['missing_images'])}")
        
        print(f"\n📋 DISTRIBUIÇÃO DE CLASSES:")
        print(f"{'Classe':<20} {'Train':<8} {'Val':<8} {'Total':<8} {'Train%':<8}")
        print("-" * 60)
        
        for i, class_name in enumerate(self.class_names):
            train_count = self.stats['train_labels'][i]
            val_count = self.stats['val_labels'][i]
            total_count = train_count + val_count
            
            if total_count > 0:
                train_percent = (train_count / total_count) * 100
                print(f"{class_name:<20} {train_count:<8} {val_count:<8} {total_count:<8} {train_percent:<8.1f}")
    
    def create_visualization(self):
        """Cria gráficos da divisão do dataset"""
        print("📊 Criando visualizações...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Gráfico de pizza - Divisão Train/Val
        sizes = [self.stats['train_images'], self.stats['val_images']]
        labels = ['Train', 'Validation']
        colors = ['#ff9999', '#66b3ff']
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('🥧 Divisão do Dataset (Train/Val)', fontsize=12, fontweight='bold')
        
        # 2. Gráfico de barras - Distribuição por classe (Train)
        classes = []
        train_counts = []
        
        for i, class_name in enumerate(self.class_names):
            if self.stats['train_labels'][i] > 0:
                classes.append(class_name)
                train_counts.append(self.stats['train_labels'][i])
        
        if train_counts:
            ax2.bar(classes, train_counts, color='lightblue', alpha=0.7)
            ax2.set_title('📊 Distribuição de Classes (Train)', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Classes')
            ax2.set_ylabel('Número de Instâncias')
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. Gráfico de barras - Distribuição por classe (Val)
        val_counts = []
        
        for i, class_name in enumerate(self.class_names):
            if self.stats['val_labels'][i] > 0:
                val_counts.append(self.stats['val_labels'][i])
        
        if val_counts and len(val_counts) == len(classes):
            ax3.bar(classes, val_counts, color='lightcoral', alpha=0.7)
            ax3.set_title('📊 Distribuição de Classes (Val)', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Classes')
            ax3.set_ylabel('Número de Instâncias')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Gráfico comparativo - Train vs Val por classe
        if train_counts and val_counts and len(train_counts) == len(val_counts):
            x = np.arange(len(classes))
            width = 0.35
            
            ax4.bar(x - width/2, train_counts, width, label='Train', color='lightblue', alpha=0.8)
            ax4.bar(x + width/2, val_counts, width, label='Val', color='lightcoral', alpha=0.8)
            
            ax4.set_title('⚖️ Comparação Train vs Val por Classe', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Classes')
            ax4.set_ylabel('Número de Instâncias')
            ax4.set_xticks(x)
            ax4.set_xticklabels(classes, rotation=45)
            ax4.legend()
        
        plt.tight_layout()
        
        # Salvar gráfico
        plot_path = self.output_path / "dataset_split_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Gráficos salvos em: {plot_path}")
    
    def export_report(self):
        """Exporta relatório detalhado da divisão"""
        report_path = self.output_path / "split_report.txt"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("RELATÓRIO DE DIVISÃO DO DATASET YOLO\n")
                f.write("="*50 + "\n\n")
                
                f.write(f"Dataset original: {self.source_path}\n")
                f.write(f"Dataset dividido: {self.output_path}\n")
                f.write(f"Proporção Train/Val: {self.train_ratio:.1%}/{self.val_ratio:.1%}\n")
                f.write(f"Seed aleatória: {self.random_seed}\n\n")
                
                f.write(f"ESTATÍSTICAS GERAIS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total de imagens: {self.stats['total_images']}\n")
                f.write(f"Imagens de treinamento: {self.stats['train_images']}\n")
                f.write(f"Imagens de validação: {self.stats['val_images']}\n\n")
                
                f.write(f"DISTRIBUIÇÃO POR CLASSE:\n")
                f.write("-" * 50 + "\n")
                f.write(f"{'Classe':<20} {'Train':<8} {'Val':<8} {'Total':<8}\n")
                f.write("-" * 50 + "\n")
                
                for i, class_name in enumerate(self.class_names):
                    train_count = self.stats['train_labels'][i]
                    val_count = self.stats['val_labels'][i]
                    total_count = train_count + val_count
                    f.write(f"{class_name:<20} {train_count:<8} {val_count:<8} {total_count:<8}\n")
                
                if self.stats['missing_labels']:
                    f.write(f"\nIMAGENS SEM LABELS:\n")
                    f.write("-" * 30 + "\n")
                    for img in self.stats['missing_labels']:
                        f.write(f"{img}\n")
                
                if self.stats['missing_images']:
                    f.write(f"\nLABELS SEM IMAGENS:\n")
                    f.write("-" * 30 + "\n")
                    for label in self.stats['missing_images']:
                        f.write(f"{label}\n")
            
            print(f"📄 Relatório salvo em: {report_path}")
            
        except Exception as e:
            print(f"⚠️  Erro ao salvar relatório: {e}")
    
    def split_dataset(self):
        """Executa a divisão completa do dataset"""
        print("🚀 Iniciando divisão do dataset YOLO...")
        print(f"📁 Origem: {self.source_path}")
        print(f"📁 Destino: {self.output_path}")
        print(f"📊 Proporções: Train {self.train_ratio:.1%}, Val {self.val_ratio:.1%}")
        
        # Validar proporções
        self.validate_ratios()
        
        # Encontrar pares imagem-label
        pairs = self.find_image_label_pairs()
        self.stats['total_images'] = len(pairs)
        
        if not pairs:
            print("❌ Nenhum par imagem-label encontrado!")
            return
        
        # Criar estrutura de diretórios
        self.create_directory_structure()
        
        # Dividir dataset de forma estratificada
        train_pairs, val_pairs = self.stratified_split(pairs)
        
        print(f"📊 Divisão realizada: {len(train_pairs)} train, {len(val_pairs)} val")
        
        # Copiar arquivos
        self.copy_files(train_pairs, "train")
        self.copy_files(val_pairs, "val")
        
        # Criar configuração YAML
        self.create_yaml_config()
        
        # Exibir estatísticas
        self.print_statistics()
        
        # Criar visualizações
        self.create_visualization()
        
        # Exportar relatório
        self.export_report()
        
        print("\n✅ Divisão do dataset concluída com sucesso!")
        print(f"🎯 Dataset pronto para treinamento YOLO em: {self.output_path}")

def main():
    """Função principal"""
    print("🤖 YOLO Dataset Splitter v1.0")
    print("="*40)
    
    # Configurações
    source_path = "Dataset"  # Seu dataset original
    output_path = "dataset"  # Dataset dividido
    
    # Configurar proporções (deve somar 1.0)
    train_ratio = 0.8  # 80% para treinamento
    val_ratio = 0.2    # 20% para validação
    
    # Criar splitter
    splitter = YOLODatasetSplitter(
        source_path=source_path,
        output_path=output_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_seed=42  # Para reprodutibilidade
    )
    
    # Executar divisão
    try:
        splitter.split_dataset()
    except Exception as e:
        print(f"❌ Erro durante a divisão: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
