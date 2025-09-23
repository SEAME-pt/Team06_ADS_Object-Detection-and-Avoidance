import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import seaborn as sns

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class YOLODatasetAnalyzer:
    def __init__(self, dataset_path, label_dirs=["labels"]):
        self.dataset_path = dataset_path
        self.label_dirs = label_dirs
        self.class_names = [
            "STOP", "PASSAGEM", "VEL_50", "VEL_80", "SEMAFORO_VERMELHO", 
            "SEMAFORO_VERDE", "SEMAFORO_LARANJA", "PASSADEIRA", "DANGER", "CURVA"
        ]
        self.num_classes = len(self.class_names)
        self.class_counts = {name: 0 for name in self.class_names}
        self.total_instances = 0
        self.total_images = 0
        self.instances_per_image = []
        
    def analyze_dataset(self):
        """Analisa o dataset e coleta estatísticas detalhadas"""
        print("🔍 Analisando dataset...")
        
        for label_dir in self.label_dirs:
            label_path = os.path.join(self.dataset_path, label_dir)
            
            if not os.path.exists(label_path):
                print(f"⚠️  Diretório não encontrado: {label_path}")
                continue
                
            for filename in os.listdir(label_path):
                if filename.endswith(".txt"):
                    file_path = os.path.join(label_path, filename)
                    instances_in_file = 0
                    
                    with open(file_path, "r") as f:
                        lines = f.readlines()
                        for line in lines:
                            if line.strip():  # Ignora linhas vazias
                                try:
                                    class_id = int(line.split()[0])
                                    if 0 <= class_id < self.num_classes:
                                        self.class_counts[self.class_names[class_id]] += 1
                                        self.total_instances += 1
                                        instances_in_file += 1
                                    else:
                                        print(f"⚠️  ID de classe inválido {class_id} em {filename}")
                                except (ValueError, IndexError):
                                    print(f"⚠️  Linha malformada em {filename}: {line.strip()}")
                    
                    self.total_images += 1
                    self.instances_per_image.append(instances_in_file)
        
        print(f"✅ Análise concluída: {self.total_images} imagens, {self.total_instances} instâncias")
    
    def calculate_balance_stats(self):
        """Calcula estatísticas de balanceamento"""
        if not self.class_counts:
            return None
            
        counts = list(self.class_counts.values())
        target_count = max(counts)
        min_count = min(counts)
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        
        missing_instances = {
            name: target_count - count 
            for name, count in self.class_counts.items()
        }
        
        # Calcular coeficiente de variação (CV)
        cv = (std_count / mean_count) * 100 if mean_count > 0 else 0
        
        return {
            'target_count': target_count,
            'min_count': min_count,
            'mean_count': mean_count,
            'std_count': std_count,
            'cv': cv,
            'missing_instances': missing_instances,
            'total_missing': sum(missing_instances.values())
        }
    
    def print_statistics(self):
        """Exibe estatísticas detalhadas"""
        stats = self.calculate_balance_stats()
        if not stats:
            print("❌ Nenhuma classe encontrada!")
            return
        
        print("\n" + "="*60)
        print("📊 ESTATÍSTICAS DO DATASET")
        print("="*60)
        
        print(f"📁 Total de imagens: {self.total_images}")
        print(f"🎯 Total de instâncias: {self.total_instances}")
        print(f"📈 Média de instâncias por imagem: {np.mean(self.instances_per_image):.2f}")
        print(f"📊 Mediana de instâncias por imagem: {np.median(self.instances_per_image):.2f}")
        
        print(f"\n🎯 BALANCEAMENTO DE CLASSES:")
        print(f"   • Classe com mais instâncias: {stats['target_count']}")
        print(f"   • Classe com menos instâncias: {stats['min_count']}")
        print(f"   • Média por classe: {stats['mean_count']:.2f}")
        print(f"   • Desvio padrão: {stats['std_count']:.2f}")
        print(f"   • Coeficiente de variação: {stats['cv']:.2f}%")
        
        if stats['cv'] > 50:
            print("   ⚠️  Dataset muito desbalanceado (CV > 50%)")
        elif stats['cv'] > 25:
            print("   ⚠️  Dataset moderadamente desbalanceado (CV > 25%)")
        else:
            print("   ✅ Dataset relativamente balanceado")
        
        print(f"\n📋 CONTAGEM POR CLASSE:")
        for name, count in sorted(self.class_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / self.total_instances) * 100
            print(f"   {name:20}: {count:4d} ({percentage:5.1f}%)")
        
        print(f"\n🔧 INSTÂNCIAS NECESSÁRIAS PARA BALANCEAMENTO:")
        print(f"   Total a adicionar: {stats['total_missing']}")
        for name, missing in sorted(stats['missing_instances'].items(), key=lambda x: x[1], reverse=True):
            if missing > 0:
                print(f"   {name:20}: +{missing:4d}")
    
    def create_visualizations(self):
        """Cria gráficos para visualizar as estatísticas"""
        if not self.class_counts:
            print("❌ Nenhum dado para visualizar!")
            return
        
        # Configurar figura com subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Gráfico de barras - Distribuição atual
        ax1 = plt.subplot(2, 3, 1)
        classes = list(self.class_counts.keys())
        counts = list(self.class_counts.values())
        
        bars = ax1.bar(classes, counts, color=plt.cm.Set3(np.linspace(0, 1, len(classes))))
        ax1.set_title('📊 Distribuição Atual de Classes', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Número de Instâncias')
        ax1.tick_params(axis='x', rotation=45)
        
        # Adicionar valores nas barras
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # 2. Gráfico de pizza - Proporção das classes
        ax2 = plt.subplot(2, 3, 2)
        wedges, texts, autotexts = ax2.pie(counts, labels=classes, autopct='%1.1f%%', 
                                          colors=plt.cm.Set3(np.linspace(0, 1, len(classes))))
        ax2.set_title('🥧 Proporção das Classes', fontsize=14, fontweight='bold')
        
        # 3. Gráfico de barras - Instâncias faltantes
        ax3 = plt.subplot(2, 3, 3)
        stats = self.calculate_balance_stats()
        missing = [stats['missing_instances'][name] for name in classes]
        
        bars3 = ax3.bar(classes, missing, color='salmon', alpha=0.7)
        ax3.set_title('🔧 Instâncias Faltantes para Balanceamento', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Classes')
        ax3.set_ylabel('Instâncias Faltantes')
        ax3.tick_params(axis='x', rotation=45)
        
        # Adicionar valores nas barras
        for bar, miss in zip(bars3, missing):
            if miss > 0:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(missing)*0.01,
                        str(miss), ha='center', va='bottom', fontweight='bold')
        
        # 4. Histograma - Instâncias por imagem
        ax4 = plt.subplot(2, 3, 4)
        ax4.hist(self.instances_per_image, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        ax4.set_title('📈 Distribuição de Instâncias por Imagem', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Número de Instâncias por Imagem')
        ax4.set_ylabel('Frequência')
        ax4.axvline(np.mean(self.instances_per_image), color='red', linestyle='--', 
                   label=f'Média: {np.mean(self.instances_per_image):.2f}')
        ax4.legend()
        
        # 5. Gráfico de barras comparativo - Atual vs Balanceado
        ax5 = plt.subplot(2, 3, 5)
        x = np.arange(len(classes))
        width = 0.35
        
        bars1 = ax5.bar(x - width/2, counts, width, label='Atual', color='lightblue', alpha=0.8)
        bars2 = ax5.bar(x + width/2, [stats['target_count']] * len(classes), width, 
                       label='Balanceado', color='lightgreen', alpha=0.8)
        
        ax5.set_title('⚖️ Comparação: Atual vs Balanceado', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Classes')
        ax5.set_ylabel('Número de Instâncias')
        ax5.set_xticks(x)
        ax5.set_xticklabels(classes, rotation=45)
        ax5.legend()
        
        # 6. Box plot - Distribuição das contagens
        ax6 = plt.subplot(2, 3, 6)
        ax6.boxplot(counts, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightcoral', alpha=0.7))
        ax6.set_title('📦 Distribuição das Contagens de Classes', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Número de Instâncias')
        ax6.set_xticklabels(['Todas as Classes'])
        
        # Adicionar estatísticas no box plot
        ax6.text(1.1, np.median(counts), f'Mediana: {np.median(counts):.0f}', 
                fontsize=10, verticalalignment='center')
        ax6.text(1.1, np.mean(counts), f'Média: {np.mean(counts):.0f}', 
                fontsize=10, verticalalignment='center')
        
        plt.tight_layout()
        plt.show()
        
        # Salvar gráfico
        try:
            fig.savefig(os.path.join(self.dataset_path, 'dataset_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"💾 Gráfico salvo em: {os.path.join(self.dataset_path, 'dataset_analysis.png')}")
        except Exception as e:
            print(f"⚠️  Erro ao salvar gráfico: {e}")
    
    def generate_augmentation_suggestions(self):
        """Gera sugestões de data augmentation"""
        stats = self.calculate_balance_stats()
        if not stats:
            return
        
        print("\n" + "="*60)
        print("💡 SUGESTÕES DE DATA AUGMENTATION")
        print("="*60)
        
        # Classes que precisam de mais dados
        classes_need_data = [(name, missing) for name, missing in stats['missing_instances'].items() if missing > 0]
        classes_need_data.sort(key=lambda x: x[1], reverse=True)
        
        if classes_need_data:
            print("🎯 Prioridade para augmentation:")
            for i, (class_name, missing) in enumerate(classes_need_data[:5], 1):
                percentage_missing = (missing / stats['target_count']) * 100
                print(f"   {i}. {class_name}: {missing} instâncias ({percentage_missing:.1f}% do target)")
        
        print("\n🔧 Técnicas de augmentation recomendadas:")
        print("   • Rotação (±15°)")
        print("   • Flip horizontal")
        print("   • Ajuste de brilho (±20%)")
        print("   • Ajuste de contraste (±20%)")
        print("   • Ruído gaussiano leve")
        print("   • Zoom (0.8x - 1.2x)")
        
        if stats['cv'] > 50:
            print("\n⚠️  Dataset muito desbalanceado!")
            print("   Considere usar weighted sampling durante o treinamento.")
    
    def export_report(self):
        """Exporta relatório detalhado"""
        stats = self.calculate_balance_stats()
        if not stats:
            return
        
        report_path = os.path.join(self.dataset_path, 'dataset_report.txt')
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("RELATÓRIO DE ANÁLISE DO DATASET YOLO\n")
                f.write("="*50 + "\n\n")
                
                f.write(f"Dataset: {self.dataset_path}\n")
                f.write(f"Total de imagens: {self.total_images}\n")
                f.write(f"Total de instâncias: {self.total_instances}\n")
                f.write(f"Média de instâncias por imagem: {np.mean(self.instances_per_image):.2f}\n\n")
                
                f.write("CONTAGEM POR CLASSE:\n")
                f.write("-" * 30 + "\n")
                for name, count in self.class_counts.items():
                    percentage = (count / self.total_instances) * 100
                    f.write(f"{name:20}: {count:4d} ({percentage:5.1f}%)\n")
                
                f.write(f"\nINSTÂNCIAS FALTANTES:\n")
                f.write("-" * 30 + "\n")
                for name, missing in stats['missing_instances'].items():
                    f.write(f"{name:20}: {missing:4d}\n")
                
                f.write(f"\nESTATÍSTICAS DE BALANCEAMENTO:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Coeficiente de variação: {stats['cv']:.2f}%\n")
                f.write(f"Desvio padrão: {stats['std_count']:.2f}\n")
                f.write(f"Total de instâncias a adicionar: {stats['total_missing']}\n")
            
            print(f"📄 Relatório salvo em: {report_path}")
            
        except Exception as e:
            print(f"⚠️  Erro ao salvar relatório: {e}")

def main():
    # Configurações
    dataset_path = "/home/djoker/Label/bin/Dataset"
    label_dirs = ["labels"]
    
    # Criar analyzer
    analyzer = YOLODatasetAnalyzer(dataset_path, label_dirs)
    
    # Executar análise
    analyzer.analyze_dataset()
    analyzer.print_statistics()
    analyzer.create_visualizations()
    analyzer.generate_augmentation_suggestions()
    analyzer.export_report()

if __name__ == "__main__":
    main()
