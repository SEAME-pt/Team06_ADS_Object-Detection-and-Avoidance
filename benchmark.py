#!/usr/bin/env python3
 
import time
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path

class InferenceSizeTester:
    def __init__(self, model_path, test_images_dir):
        """
        Args:
            model_path: Caminho para o modelo treinado (.pt)
            test_images_dir: Pasta com imagens de teste
        """
        self.model = YOLO(model_path)
        self.test_images = list(Path(test_images_dir).glob("*.jpg")) + \
                          list(Path(test_images_dir).glob("*.png"))
        
        # Configurações de teste
        self.test_sizes = [224, 320, 416, 512, 640, 768, 896]
        self.confidence_threshold = 0.5
        
    def benchmark_size(self, imgsz, num_runs=10):
        """Testa um tamanho específico"""
        print(f"\n🔍 Testando tamanho: {imgsz}x{imgsz}")
        
        times = []
        all_detections = []
        
        for i, img_path in enumerate(self.test_images[:num_runs]):
            # Carregar imagem
            image = cv2.imread(str(img_path))
            
            # Medir tempo de inferência
            start_time = time.time()
            results = self.model(image, imgsz=imgsz, conf=self.confidence_threshold, verbose=False)
            inference_time = time.time() - start_time
            
            times.append(inference_time)
            all_detections.append(len(results[0].boxes) if results[0].boxes is not None else 0)
            
            if (i + 1) % 5 == 0:
                print(f"  Processadas {i+1}/{num_runs} imagens...")
        
        avg_time = np.mean(times)
        fps = 1.0 / avg_time
        avg_detections = np.mean(all_detections)
        
        return {
            'size': imgsz,
            'avg_time': avg_time,
            'fps': fps,
            'avg_detections': avg_detections,
            'times': times
        }
    
    def run_benchmark(self, num_runs=10):
        """Executa benchmark completo"""
        print("🚀 Iniciando benchmark de tamanhos de inferência...")
        print(f"📊 Testando {len(self.test_images)} imagens com {num_runs} runs cada")
        
        results = []
        
        for size in self.test_sizes:
            try:
                result = self.benchmark_size(size, num_runs)
                results.append(result)
                
                print(f"✅ {size}x{size}: {result['fps']:.1f} FPS, "
                      f"{result['avg_time']*1000:.1f}ms, "
                      f"{result['avg_detections']:.1f} detecções")
                      
            except Exception as e:
                print(f"❌ Erro no tamanho {size}: {e}")
        
        return results
    
    def plot_results(self, results):
        """Plota gráficos comparativos"""
        sizes = [r['size'] for r in results]
        fps_values = [r['fps'] for r in results]
        times = [r['avg_time']*1000 for r in results]  # em ms
        detections = [r['avg_detections'] for r in results]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # FPS vs Tamanho
        ax1.plot(sizes, fps_values, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Tamanho da Imagem (pixels)')
        ax1.set_ylabel('FPS')
        ax1.set_title('FPS vs Tamanho da Imagem')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(sizes)
        
        # Tempo vs Tamanho
        ax2.plot(sizes, times, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Tamanho da Imagem (pixels)')
        ax2.set_ylabel('Tempo (ms)')
        ax2.set_title('Tempo de Inferência vs Tamanho')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(sizes)
        
        # Detecções vs Tamanho
        ax3.plot(sizes, detections, 'go-', linewidth=2, markersize=8)
        ax3.set_xlabel('Tamanho da Imagem (pixels)')
        ax3.set_ylabel('Número Médio de Detecções')
        ax3.set_title('Detecções vs Tamanho da Imagem')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(sizes)
        
        # Trade-off FPS vs Detecções
        ax4.scatter(fps_values, detections, c=sizes, cmap='viridis', s=100)
        ax4.set_xlabel('FPS')
        ax4.set_ylabel('Número Médio de Detecções')
        ax4.set_title('Trade-off: Velocidade vs Detecções')
        ax4.grid(True, alpha=0.3)
        
        # Adicionar labels nos pontos do scatter
        for i, size in enumerate(sizes):
            ax4.annotate(f'{size}px', (fps_values[i], detections[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        plt.savefig('inference_benchmark.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def get_recommendations(self, results):
        """Gera recomendações baseadas nos resultados"""
        print("\n RECOMENDAÇÕES:")
        print("=" * 50)
        
        # Melhor FPS
        best_fps = max(results, key=lambda x: x['fps'])
        print(f" Melhor Performance: {best_fps['size']}px - {best_fps['fps']:.1f} FPS")
        
        # Melhor detecção
        best_detection = max(results, key=lambda x: x['avg_detections'])
        print(f" Melhor Detecção: {best_detection['size']}px - {best_detection['avg_detections']:.1f} detecções")
        
        # Balance (FPS > 20 com boa detecção)
        balanced = [r for r in results if r['fps'] > 20]
        if balanced:
            best_balanced = max(balanced, key=lambda x: x['avg_detections'])
            print(f"⚖️ Melhor Balance: {best_balanced['size']}px - {best_balanced['fps']:.1f} FPS, {best_balanced['avg_detections']:.1f} detecções")
        
        # Recomendações por uso
        print(f"\n RECOMENDAÇÕES POR USO:")
        print(f"• Jetson Nano (tempo real): {[r['size'] for r in results if r['fps'] > 15][0] if any(r['fps'] > 15 for r in results) else 'N/A'}px")
        print(f"• Aplicação crítica: {best_detection['size']}px")
        print(f"• Streaming/webcam: {[r['size'] for r in results if r['fps'] > 25][0] if any(r['fps'] > 25 for r in results) else 'N/A'}px")


def main():
    MODEL_PATH = "best.pt" 
    TEST_IMAGES_DIR = "dataset/images/val"  # Imagens de validação
    
    tester = InferenceSizeTester(MODEL_PATH, TEST_IMAGES_DIR)
    results = tester.run_benchmark(num_runs=20)  # 20 imagens de teste
    tester.plot_results(results)
    tester.get_recommendations(results)
    
    print(f"\n RESULTADOS DETALHADOS:")
    print("-" * 70)
    print(f"{'Tamanho':<8} {'FPS':<8} {'Tempo(ms)':<12} {'Detecções':<12} {'Recomendação'}")
    print("-" * 70)
    
    for result in results:
        size = result['size']
        fps = result['fps']
        time_ms = result['avg_time'] * 1000
        detections = result['avg_detections']
        
        # Classificação
        if fps > 30:
            rec = " Rápido"
        elif fps > 20:
            rec = " Balanceado"
        elif fps > 10:
            rec = " Precisão"
        else:
            rec = " Lento"
            
        print(f"{size:<8} {fps:<8.1f} {time_ms:<12.1f} {detections:<12.1f} {rec}")


if __name__ == "__main__":
    main()
