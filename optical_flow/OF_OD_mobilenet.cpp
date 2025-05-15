#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <vector>
#include <string>
#include <iostream>

using namespace cv;
using namespace cv::cuda;
using namespace std;

// Função para desenhar vetores de fluxo óptico
void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, double, const Scalar& color) {
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, Point(x,y), 2, color, -1);
        }
}

int main() {
    // Inicializar câmera CSI (IMX219-160)
    VideoCapture cap("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink");
    if (!cap.isOpened()) {
        cerr << "Erro ao abrir a câmera" << endl;
        return -1;
    }

    // Carregar modelo MobileNet SSD
    String modelConfiguration = "ssd_mobilenet_v2_coco.pbtxt";
    String modelWeights = "ssd_mobilenet_v2_coco.pb";
    dnn::Net net = dnn::readNetFromTensorflow(modelWeights, modelConfiguration);
    net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(dnn::DNN_TARGET_CUDA);

    // Classes do modelo COCO
    vector<string> classes = {"background", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
                              "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                              "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                              "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
                              "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
                              "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                              "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                              "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
                              "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

    // Inicializar Optical Flow com CUDA
    Ptr<cuda::DenseOpticalFlow> optflow = cuda::DenseOpticalFlow::createFarneback();

    Mat frame, prev_frame, flow_cpu;
    GpuMat frame_gpu, prev_frame_gpu, flow_gpu;

    // Capturar primeiro frame
    cap >> frame;
    if (frame.empty()) {
        cerr << "Frame vazio" << endl;
        return -1;
    }
    cvtColor(frame, prev_frame, COLOR_BGR2GRAY);
    prev_frame_gpu.upload(prev_frame);

    while (true) {
        // Capturar novo frame
        cap >> frame;
        if (frame.empty())
            break;

        // Converter para escala de cinza e fazer upload para GPU
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        frame_gpu.upload(gray);

        // Calcular Optical Flow na GPU
        optflow->calc(prev_frame_gpu, frame_gpu, flow_gpu);
        flow_gpu.download(flow_cpu);

        // Desenhar vetores de fluxo óptico
        Mat cflow;
        cvtColor(gray, cflow, COLOR_GRAY2BGR);
        drawOptFlowMap(flow_cpu, cflow, 16, 1.5, Scalar(0, 255, 0));

        // Detectar regiões com movimento significativo
        Mat flow_mag;
        vector<Mat> flow_xy;
        split(flow_cpu, flow_xy);
        magnitude(flow_xy[0], flow_xy[1], flow_mag);
        threshold(flow_mag, flow_mag, 2.0, 255, THRESH_BINARY);
        flow_mag.convertTo(flow_mag, CV_8U);

        // Encontrar contornos de regiões com movimento
        vector<vector<Point>> contours;
        findContours(flow_mag, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // Preparar frame para detecção de objetos
        Mat blob = dnn::blobFromImage(frame, 1.0, Size(300, 300), Scalar(127.5, 127.5, 127.5), true, false);
        net.setInput(blob);
        Mat detections = net.forward();

        // Processar detecções apenas em regiões com movimento
        for (size_t i = 0; i < contours.size(); i++) {
            Rect roi = boundingRect(contours[i]);
            for (int j = 0; j < detections.size[2]; j++) {
                float confidence = detections.at<float>(0, 0, j, 2);
                if (confidence > 0.5) {
                    int classId = (int)(detections.at<float>(0, 0, j, 1));
                    float x1 = detections.at<float>(0, 0, j, 3) * frame.cols;
                    float y1 = detections.at<float>(0, 0, j, 4) * frame.rows;
                    float x2 = detections.at<float>(0, 0, j, 5) * frame.cols;
                    float y2 = detections.at<float>(0, 0, j, 6) * frame.rows;
                    Rect box(x1, y1, x2 - x1, y2 - y1);

                    // Verificar se a detecção está dentro da região de movimento
                    if ((box & roi).area() > 0) {
                        rectangle(cflow, box, Scalar(0, 0, 255), 2);
                        putText(cflow, classes[classId], Point(x1, y1 - 10),
                                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
                    }
                }
            }
        }

        // Mostrar resultado
        imshow("Optical Flow e Detecção", cflow);
        if (waitKey(1) == 27) // Pressione ESC para sair
            break;

        // Atualizar frame anterior
        prev_frame_gpu = frame_gpu.clone();
    }

    cap.release();
    destroyAllWindows();
    return 0;
}