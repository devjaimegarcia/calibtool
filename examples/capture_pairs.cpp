#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <cstdio>
#include <unistd.h>
#include <cstdlib>
#include <cstring>

static std::string gstCam(int sensorId, int w, int h, int fps) {
    return "nvarguscamerasrc sensor-id=" + std::to_string(sensorId) + " ! "
           "video/x-raw(memory:NVMM),width=" + std::to_string(w) +
           ",height=" + std::to_string(h) +
           ",framerate=" + std::to_string(fps) + "/1,format=NV12 ! "
           "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
           "video/x-raw,format=BGR ! "
           "appsink max-buffers=1 drop=true sync=false";
}

static std::string gstCamGray(int sensorId, int w, int h, int fps) {
    return "nvarguscamerasrc sensor-id=" + std::to_string(sensorId) + " ! "
           "video/x-raw(memory:NVMM),width=" + std::to_string(w) +
           ",height=" + std::to_string(h) +
           ",framerate=" + std::to_string(fps) + "/1,format=NV12 ! "
           "nvvidconv ! video/x-raw,format=GRAY8 ! "
           "appsink max-buffers=1 drop=true sync=false";
}

int main() {
    const int W=1280, H=720, FPS=30;
    const cv::Size boardSize(8,8);            // esquinas internas 10x7
    const std::string outDir = "stereo_pairs";

    std::filesystem::create_directories(outDir);

    cv::VideoCapture capL(gstCamGray(0,W,H,FPS), cv::CAP_GSTREAMER);
    cv::VideoCapture capR(gstCamGray(1,W,H,FPS), cv::CAP_GSTREAMER);
    if(!capL.isOpened() || !capR.isOpened()){
        std::cerr << "No pude abrir cámaras (sensor-id 0/1)\n";
        return 1;
    }

    int idx = 0;
    bool fL = false;
    bool fR = false;

    while(true){
        cv::Mat L, R;
        if(!capL.read(L) || !capR.read(R)) continue;

        // cv::Mat gL, gR;
        // cv::cvtColor(L, gL, cv::COLOR_BGR2GRAY);
        // cv::cvtColor(R, gR, cv::COLOR_BGR2GRAY);


        std::vector<cv::Point2f> cL, cR;

        fL = cv::findChessboardCorners(L, boardSize, cL, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
        fR = cv::findChessboardCorners(R, boardSize, cR, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

        if(fL){
            cv::cornerSubPix(L, cL, {11,11}, {-1,-1},
                             cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 30, 0.01));
            cv::drawChessboardCorners(L, boardSize, cL, fL);
        }
        if(fR){
            cv::cornerSubPix(R, cR, {11,11}, {-1,-1},
                             cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 30, 0.01));
            cv::drawChessboardCorners(R, boardSize, cR, fR);
        }

        cv::Mat show;
        cv::hconcat(L, R, show);
        cv::putText(show, "S=save pair  Q=quit", {20,40}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,255,0}, 2);
        cv::imshow("stereo capture", show);

        int k = cv::waitKey(1);
        if(k=='q' || k=='Q') break;

        if((k=='s' || k=='S') && fL && fR){
            char nameL[128], nameR[128];
            std::snprintf(nameL, sizeof(nameL), "%s/L_%04d.png", outDir.c_str(), idx);
            std::snprintf(nameR, sizeof(nameR), "%s/R_%04d.png", outDir.c_str(), idx);
            // cv::imwrite(nameL, gL);
            // cv::imwrite(nameR, gR);
            std::cout << "Saved pair " << idx << "\n";
            idx++;
        }
    }
    return 0;
}
