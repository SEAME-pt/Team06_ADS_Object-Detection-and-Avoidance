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
void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, const Scalar& color) {
    for (int y = 0; y < cflowmap.rows; y += step)
        for (int x = 0; x < cflowmap.cols; x += step) {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x, y), Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), color);
            circle(cflowmap, Point(x, y), 2, color, -1);
        }
}

int main() {
    // Inicializar câmera CSI (IMX219-160) com modo 4
    VideoCapture cap("nvarguscamerasrc sensor-mode=4 ! video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! nvvidconv ! video/x-raw, width=416, height=416, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink");
    if (!cap.isOpened()) {
        cerr << "Erro ao abrir a câmera" << endl;
        return -1;
    }

    // Inicializar Optical Flow com CUDA
    Ptr<cuda::FarnebackOpticalFlow> optflow = cuda::FarnebackOpticalFlow::create();

    Mat frame, prev_frame, flow_cpu;
    GpuMat frame_gpu, prev_frame_gpu, flow_gpu;

    // Capturar primeiro frame
    cap >> frame;
    if (frame.empty()) {
        cerr << "Frame vazio" << endl;
        return -1;
    }
    cv::cvtColor(frame, prev_frame, COLOR_BGR2GRAY);
    prev_frame_gpu.upload(prev_frame);

    while (true) {
        // Capturar novo frame
        cap >> frame;
        if (frame.empty())
            break;

        // Converter para escala de cinza e fazer upload para GPU
        Mat gray;
        cv::cvtColor(frame, gray, COLOR_BGR2GRAY);
        frame_gpu.upload(gray);

        // Calcular Optical Flow na GPU
        optflow->calc(prev_frame_gpu, frame_gpu, flow_gpu);
        flow_gpu.download(flow_cpu);

        // Desenhar vetores de fluxo óptico
        Mat cflow;
        cv::cvtColor(frame, cflow, COLOR_BGR2RGB); // Usar frame colorido para melhor visualização
        drawOptFlowMap(flow_cpu, cflow, 16, Scalar(0, 255, 0));

        // Calcular magnitude do fluxo para detectar movimento
        Mat flow_mag;
        vector<Mat> flow_xy;
        split(flow_cpu, flow_xy);
        magnitude(flow_xy[0], flow_xy[1], flow_mag);
        threshold(flow_mag, flow_mag, 3.0, 255, THRESH_BINARY); // Ajustar threshold para sensibilidade
        flow_mag.convertTo(flow_mag, CV_8U);

        // Encontrar contornos de regiões com movimento
        vector<vector<Point>> contours;
        findContours(flow_mag, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // Desenhar caixas delimitadoras ao redor dos contornos
        for (size_t i = 0; i < contours.size(); i++) {
            Rect roi = boundingRect(contours[i]);
            if (roi.area() > 100) { // Filtrar contornos pequenos (ruído)
                rectangle(cflow, roi, Scalar(0, 0, 255), 2);
                putText(cflow, "Objeto", Point(roi.x, roi.y - 10),
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