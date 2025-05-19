#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <vector>
#include <iostream>

using namespace cv;
using namespace cv::cuda;
using namespace std;

// Função para desenhar vetores de fluxo óptico
void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, const Scalar& color, const Rect& roi) {
    for (int y = roi.y; y < roi.y + roi.height; y += step)
        for (int x = roi.x; x < roi.x + roi.width; x += step) {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x, y), Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), color);
            circle(cflowmap, Point(x, y), 2, color, -1);
        }
}

int main() {
    // Inicializar câmera CSI (IMX219-160) com modo 4, 640x480
    VideoCapture cap("nvarguscamerasrc sensor-mode=4 ! video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink");
    if (!cap.isOpened()) {
        cerr << "Erro ao abrir a câmera" << endl;
        return -1;
    }

    // Definir dimensões da imagem redimensionada
    int width = 640, height = 320;
    Size targetSize(width, height);

    // Definir ROI (metade inferior da imagem redimensionada)
    Rect roi(0, height / 2, width, height / 2); // 640x160

    // Inicializar Optical Flow com CUDA e parâmetros otimizados
    Ptr<cuda::FarnebackOpticalFlow> optflow = cuda::FarnebackOpticalFlow::create(
        3,  // numLevels: menos níveis para maior velocidade
        0.5, // pyrScale
        false, // fastPyramids
        15,  // winSize: maior janela para suavização
        3,   // numIters: menos iterações
        5,   // polyN
        1.2  // polySigma
    );

    Mat frame, frame_resized, prev_frame, flow_cpu;
    GpuMat frame_gpu, prev_frame_gpu, flow_gpu;

    // Capturar primeiro frame
    cap >> frame;
    if (frame.empty()) {
        cerr << "Frame vazio na inicialização" << endl;
        return -1;
    }
    // Redimensionar para 640x320
    cv::resize(frame, frame_resized, targetSize);
    Mat frame_roi = frame_resized(roi);
    Mat prev_frame_roi;
    cv::cvtColor(frame_roi, prev_frame_roi, COLOR_BGR2GRAY);
    prev_frame_gpu.upload(prev_frame_roi);

    while (true) {
        // Capturar novo frame
        cap >> frame;
        if (frame.empty()) {
            cerr << "Frame vazio durante o loop" << endl;
            break;
        }

        // Redimensionar para 640x320
        cv::resize(frame, frame_resized, targetSize);

        // Extrair ROI do frame redimensionado
        Mat frame_roi = frame_resized(roi);
        Mat gray_roi;
        cv::cvtColor(frame_roi, gray_roi, COLOR_BGR2GRAY);
        frame_gpu.upload(gray_roi);

        // Calcular Optical Flow na GPU (apenas na ROI)
        optflow->calc(prev_frame_gpu, frame_gpu, flow_gpu);
        flow_gpu.download(flow_cpu);

        // Desenhar vetores de fluxo óptico
        Mat cflow = frame_resized.clone(); // Usar frame redimensionado para exibição
        drawOptFlowMap(flow_cpu, cflow, 24, Scalar(0, 255, 0), roi);

        // Calcular magnitude do fluxo para detectar movimento
        Mat flow_mag;
        vector<Mat> flow_xy;
        split(flow_cpu, flow_xy);
        magnitude(flow_xy[0], flow_xy[1], flow_mag);
        threshold(flow_mag, flow_mag, 3.0, 255, THRESH_BINARY);
        flow_mag.convertTo(flow_mag, CV_8U);

        // Encontrar contornos de regiões com movimento
        vector<vector<Point>> contours;
        findContours(flow_mag, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // Desenhar caixas delimitadoras ao redor dos contornos, ajustadas para o frame completo
        for (size_t i = 0; i < contours.size(); i++) {
            Rect roi_rel = boundingRect(contours[i]);
            if (roi_rel.area() > 50) { // Reduzido para nova resolução
                // Ajustar coordenadas para o frame redimensionado
                Rect roi_abs(roi_rel.x + roi.x, roi_rel.y + roi.y, roi_rel.width, roi_rel.height);
                rectangle(cflow, roi_abs, Scalar(0, 0, 255), 2);
                putText(cflow, "Objeto", Point(roi_abs.x, roi_abs.y - 10),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
            }
        }

        // Mostrar resultado
        imshow("Optical Flow - Detecção de Objetos", cflow);
        if (waitKey(1) == 27) // Pressione ESC para sair
            break;

        // Atualizar frame anterior
        prev_frame_gpu = frame_gpu.clone();
    }

    cap.release();
    destroyAllWindows();
    return 0;
}