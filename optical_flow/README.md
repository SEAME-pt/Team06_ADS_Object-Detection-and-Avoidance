g++ -o optical_flow_detectionv2 OF_OD_v2.cpp \
-I/usr/local/include/opencv4 \
-L/usr/local/lib \
-lopencv_core -lopencv_imgproc -lopencv_video -lopencv_highgui \
-lopencv_cudaoptflow -lopencv_cudawarping -lopencv_cudaimgproc \
-lopencv_imgcodecs -lopencv_videoio

