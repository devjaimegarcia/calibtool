#include <opencv2/opencv.hpp>

#include <vpi/Stream.h>
#include <vpi/Image.h>
#include <vpi/OpenCVInterop.hpp>

#include <vpi/algo/StereoDisparity.h>
#include <vpi/algo/ConvertImageFormat.h>

#include <iostream>
#include <stdexcept>

#define VPI_CHECK(call) do {                          \
    VPIStatus _st = (call);                           \
    if (_st != VPI_SUCCESS) {                         \
        char msg[VPI_MAX_STATUS_MESSAGE_LENGTH];      \
        vpiGetLastStatusMessage(msg, sizeof(msg));    \
        throw std::runtime_error(msg);                \
    }                                                 \
} while(0)

static std::string gstCamGray(int sensorId, int w, int h, int fps) {
    return "nvarguscamerasrc sensor-id=" + std::to_string(sensorId) + " ! "
           "video/x-raw(memory:NVMM),width=" + std::to_string(w) +
           ",height=" + std::to_string(h) +
           ",framerate=" + std::to_string(fps) + "/1,format=NV12 ! "
           "nvvidconv ! video/x-raw,format=GRAY8 ! "
           "appsink max-buffers=1 drop=true sync=false";
}

int main() {
    const int W = 640, H = 360, FPS = 15;          // empieza con esto
    const uint64_t BACKEND = VPI_BACKEND_CUDA;     // luego puedes probar OFA/OFA+PVA+VIC

    // Captura GRAY8
    cv::VideoCapture capL(gstCamGray(0, W, H, FPS), cv::CAP_GSTREAMER);
    cv::VideoCapture capR(gstCamGray(1, W, H, FPS), cv::CAP_GSTREAMER);
    if(!capL.isOpened() || !capR.isOpened()) {
        std::cerr << "No pude abrir CSI sensor-id 0/1\n";
        return 1;
    }

    // VPI init
    VPIStream stream;
    VPI_CHECK(vpiStreamCreate(0, &stream));

    // Imágenes VPI persistentes (reusables)
    VPIImage l16=nullptr, r16=nullptr, dispS16=nullptr, confU16=nullptr, dispU8=nullptr;

    VPI_CHECK(vpiImageCreate(W, H, VPI_IMAGE_FORMAT_U16, 0, &l16));
    VPI_CHECK(vpiImageCreate(W, H, VPI_IMAGE_FORMAT_U16, 0, &r16));
    VPI_CHECK(vpiImageCreate(W, H, VPI_IMAGE_FORMAT_S16, 0, &dispS16)); // disparity output típico :contentReference[oaicite:6]{index=6}
    VPI_CHECK(vpiImageCreate(W, H, VPI_IMAGE_FORMAT_U16, 0, &confU16));
    VPI_CHECK(vpiImageCreate(W, H, VPI_IMAGE_FORMAT_U8,  0, &dispU8));  // para display

    // Payload stereo
    VPIPayload stereo;
    VPI_CHECK(vpiCreateStereoDisparityEstimator(BACKEND, W, H, VPI_IMAGE_FORMAT_U16, nullptr, &stereo)); // :contentReference[oaicite:7]{index=7}

    // Params
    VPIStereoDisparityEstimatorParams sparams;
    VPI_CHECK(vpiInitStereoDisparityEstimatorParams(&sparams));
    sparams.windowSize   = 5;
    sparams.maxDisparity = 64;

    // Convert params: U8->U16 (input)
    VPIConvertImageFormatParams inCvt;
    VPI_CHECK(vpiInitConvertImageFormatParams(&inCvt));
    inCvt.scale = 1.0f;

    // Convert params: S16 disparity -> U8 display (igual que doc)
    VPIConvertImageFormatParams outCvt;
    VPI_CHECK(vpiInitConvertImageFormatParams(&outCvt));
    outCvt.scale = 1.0f / (32.0f * sparams.maxDisparity) * 255.0f; // :contentReference[oaicite:8]{index=8}

    cv::Mat L8, R8;
    while(true) {
        if(!capL.read(L8) || !capR.read(R8)) continue;
        if(L8.empty() || R8.empty()) continue;

        // Wrap OpenCV Mat -> VPIImage (sin copiar)
        VPIImage l8wrap=nullptr, r8wrap=nullptr;
        VPI_CHECK(vpiImageCreateWrapperOpenCVMat(L8, VPI_IMAGE_FORMAT_U8, 0, &l8wrap)); // :contentReference[oaicite:9]{index=9}
        VPI_CHECK(vpiImageCreateWrapperOpenCVMat(R8, VPI_IMAGE_FORMAT_U8, 0, &r8wrap));

        // U8 -> U16 (para alimentar estimator como en docs)
        VPI_CHECK(vpiSubmitConvertImageFormat(stream, BACKEND, l8wrap, l16, &inCvt));
        VPI_CHECK(vpiSubmitConvertImageFormat(stream, BACKEND, r8wrap, r16, &inCvt));

        // Stereo disparity
        VPI_CHECK(vpiSubmitStereoDisparityEstimator(stream, BACKEND, stereo, l16, r16, dispS16, confU16, &sparams)); // :contentReference[oaicite:10]{index=10}

        // Disparity -> U8 para mostrar
        VPI_CHECK(vpiSubmitConvertImageFormat(stream, BACKEND, dispS16, dispU8, &outCvt)); // :contentReference[oaicite:11]{index=11}

        VPI_CHECK(vpiStreamSync(stream));

        // Export VPI -> OpenCV Mat para imshow
        VPIImageData data;
        VPI_CHECK(vpiImageLockData(dispU8, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data)); // :contentReference[oaicite:12]{index=12}
        cv::Mat dispMat;
        VPI_CHECK(vpiImageDataExportOpenCVMat(data, &dispMat)); // :contentReference[oaicite:13]{index=13}
        VPI_CHECK(vpiImageUnlock(dispU8));

        cv::imshow("Right (GRAY)", R8);
        cv::imshow("Left (GRAY)", L8);
        cv::imshow("Disparity (U8)", dispMat);

        int k = cv::waitKey(1);
        if(k=='q' || k=='Q') break;

        vpiImageDestroy(l8wrap);
        vpiImageDestroy(r8wrap);
    }

    vpiPayloadDestroy(stereo);
    vpiImageDestroy(l16); vpiImageDestroy(r16);
    vpiImageDestroy(dispS16); vpiImageDestroy(confU16); vpiImageDestroy(dispU8);
    vpiStreamDestroy(stream);
    return 0;
}
