#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

static bool findCornersRobust(const cv::Mat &gray, const cv::Size &boardSize,
                              std::vector<cv::Point2f> &corners)
{
    corners.clear();

#if CV_VERSION_MAJOR >= 4
    // findChessboardCornersSB suele ser más robusto si está disponible
    bool ok = cv::findChessboardCornersSB(gray, boardSize, corners,
                                          cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY);
    if(ok) 
        return true;
#else
    bool ok = false;
#endif

    // Fallback clásico
    ok = cv::findChessboardCorners(gray, boardSize, corners,
                cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
    if(!ok) return false;

    cv::cornerSubPix(gray, corners, cv::Size(11,11), cv::Size(-1,-1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
    return true;
}

static std::vector<std::string> listSorted(const std::string &dir, const std::string &prefix)
{
    std::vector<std::string> files;
    for(const auto &e : fs::directory_iterator(dir)) {
        if(!e.is_regular_file()) continue;
        auto p = e.path().filename().string();
        if(p.rfind(prefix, 0) == 0) { // starts_with prefix
            files.push_back(e.path().string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

int main(int argc, char** argv)
{
    const std::string pairsDir = (argc > 1) ? argv[1] : "stereo_pairs";
    const std::string outYml   = (argc > 2) ? argv[2] : "stereo.yml";

    const cv::Size boardSize(10, 7);     // esquinas internas (10x7)
    const double squareSize = 0.05;      // 5 cm = 0.05 m

    auto leftFiles  = listSorted(pairsDir, "L_");
    auto rightFiles = listSorted(pairsDir, "R_");

    if(leftFiles.empty() || rightFiles.empty()) {
        std::cerr << "No encontre imagenes en " << pairsDir << " (esperaba L_####.png y R_####.png)\n";
        return 1;
    }
    if(leftFiles.size() != rightFiles.size()) {
        std::cerr << "Cantidad distinta: L=" << leftFiles.size()
                  << " R=" << rightFiles.size() << "\n";
        std::cerr << "Asegurate que existan pares L_#### y R_#### para el mismo indice.\n";
        return 1;
    }

    cv::Size imageSize;
    std::vector<std::vector<cv::Point2f>> imgPtsL, imgPtsR;
    std::vector<std::vector<cv::Point3f>> objPts;

    // Patrón 3D en el plano Z=0
    std::vector<cv::Point3f> obj;
    obj.reserve(boardSize.width * boardSize.height);
    for(int y=0; y<boardSize.height; ++y)
        for(int x=0; x<boardSize.width; ++x)
            obj.emplace_back(float(x * squareSize), float(y * squareSize), 0.0f);

    std::cout << "Leyendo " << leftFiles.size() << " pares...\n";
    int used = 0;

    for(size_t i=0; i<leftFiles.size(); ++i) {
        cv::Mat gL = cv::imread(leftFiles[i], cv::IMREAD_GRAYSCALE);
        cv::Mat gR = cv::imread(rightFiles[i], cv::IMREAD_GRAYSCALE);
        if(gL.empty() || gR.empty()) {
            std::cerr << "No pude leer: " << leftFiles[i] << " o " << rightFiles[i] << "\n";
            continue;
        }
        if(imageSize.width == 0) imageSize = gL.size();
        if(gL.size() != imageSize || gR.size() != imageSize) {
            std::cerr << "Tamanio inconsistente en par " << i << "\n";
            continue;
        }

        std::vector<cv::Point2f> cL, cR;
        bool fL = findCornersRobust(gL, boardSize, cL);
        bool fR = findCornersRobust(gR, boardSize, cR);

        // Vista rápida
        cv::Mat vis;
        cv::cvtColor(gL, vis, cv::COLOR_GRAY2BGR);
        if(fL) cv::drawChessboardCorners(vis, boardSize, cL, fL);
        cv::imshow("Left corners", vis);

        cv::cvtColor(gR, vis, cv::COLOR_GRAY2BGR);
        if(fR) cv::drawChessboardCorners(vis, boardSize, cR, fR);
        cv::imshow("Right corners", vis);

        int k = cv::waitKey(10);
        (void)k;

        if(fL && fR) {
            imgPtsL.push_back(cL);
            imgPtsR.push_back(cR);
            objPts.push_back(obj);
            used++;
        } else {
            std::cout << "Descartando par " << i << " (corners L=" << fL << " R=" << fR << ")\n";
        }
    }

    if(used < 15) {
        std::cerr << "Muy pocos pares validos: " << used << ". Captura al menos 30-80.\n";
        return 1;
    }

    std::cout << "Pares validos: " << used << "\n";
    cv::destroyWindow("Left corners");
    cv::destroyWindow("Right corners");

    // 1) Calibración individual
    cv::Mat K1 = cv::Mat::eye(3,3,CV_64F), D1 = cv::Mat::zeros(1,8,CV_64F);
    cv::Mat K2 = cv::Mat::eye(3,3,CV_64F), D2 = cv::Mat::zeros(1,8,CV_64F);

    std::vector<cv::Mat> rvecs1, tvecs1, rvecs2, tvecs2;

    int flagsCam = 0; // puedes probar CALIB_RATIONAL_MODEL si quieres
    double err1 = cv::calibrateCamera(objPts, imgPtsL, imageSize, K1, D1, rvecs1, tvecs1, flagsCam);
    double err2 = cv::calibrateCamera(objPts, imgPtsR, imageSize, K2, D2, rvecs2, tvecs2, flagsCam);

    std::cout << "Reproj error L: " << err1 << " px\n";
    std::cout << "Reproj error R: " << err2 << " px\n";

    // 2) Calibración estéreo (fijando intrínsecos)
    cv::Mat R, T, E, F;
    int flagsStereo = cv::CALIB_FIX_INTRINSIC;
    cv::TermCriteria crit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-6);

    double stereoErr = cv::stereoCalibrate(
        objPts, imgPtsL, imgPtsR,
        K1, D1, K2, D2,
        imageSize, R, T, E, F,
        flagsStereo, crit
    );

    std::cout << "Stereo reproj error: " << stereoErr << " px\n";
    std::cout << "T (metros) = [" << T.at<double>(0) << ", " << T.at<double>(1) << ", " << T.at<double>(2) << "]\n";
    std::cout << "Baseline aprox (m) = " << cv::norm(T) << "\n";

    // 3) Rectificación
    cv::Mat R1, R2, P1, P2, Q;
    cv::Rect validRoi1, validRoi2;

    cv::stereoRectify(K1, D1, K2, D2, imageSize, R, T, R1, R2, P1, P2, Q,
                      cv::CALIB_ZERO_DISPARITY, 0, imageSize, &validRoi1, &validRoi2);

    cv::Mat map1x, map1y, map2x, map2y;
    cv::initUndistortRectifyMap(K1, D1, R1, P1, imageSize, CV_32FC1, map1x, map1y);
    cv::initUndistortRectifyMap(K2, D2, R2, P2, imageSize, CV_32FC1, map2x, map2y);

    // Guardar YAML
    {
        cv::FileStorage fs(outYml, cv::FileStorage::WRITE);
        fs << "imageWidth"  << imageSize.width;
        fs << "imageHeight" << imageSize.height;
        fs << "boardWidth"  << boardSize.width;
        fs << "boardHeight" << boardSize.height;
        fs << "squareSize_m" << squareSize;

        fs << "K1" << K1 << "D1" << D1;
        fs << "K2" << K2 << "D2" << D2;
        fs << "R"  << R  << "T"  << T;
        fs << "E"  << E  << "F"  << F;

        fs << "R1" << R1 << "R2" << R2;
        fs << "P1" << P1 << "P2" << P2;
        fs << "Q"  << Q;

        fs << "map1x" << map1x << "map1y" << map1y;
        fs << "map2x" << map2x << "map2y" << map2y;

        fs << "validRoi1_x" << validRoi1.x << "validRoi1_y" << validRoi1.y
           << "validRoi1_w" << validRoi1.width << "validRoi1_h" << validRoi1.height;
        fs << "validRoi2_x" << validRoi2.x << "validRoi2_y" << validRoi2.y
           << "validRoi2_w" << validRoi2.width << "validRoi2_h" << validRoi2.height;
    }

    std::cout << "Guardado: " << outYml << "\n";

    // 4) Verificación visual con un par (rectificado + líneas)
    {
        cv::Mat gL = cv::imread(leftFiles.back(),  cv::IMREAD_GRAYSCALE);
        cv::Mat gR = cv::imread(rightFiles.back(), cv::IMREAD_GRAYSCALE);
        cv::Mat rL, rR;
        cv::remap(gL, rL, map1x, map1y, cv::INTER_LINEAR);
        cv::remap(gR, rR, map2x, map2y, cv::INTER_LINEAR);

        cv::Mat showL, showR, show;
        cv::cvtColor(rL, showL, cv::COLOR_GRAY2BGR);
        cv::cvtColor(rR, showR, cv::COLOR_GRAY2BGR);
        cv::hconcat(showL, showR, show);

        // líneas horizontales para checar rectificación
        for(int y=0; y<show.rows; y+=40) {
            cv::line(show, cv::Point(0,y), cv::Point(show.cols-1,y), cv::Scalar(0,255,0), 1);
        }
        cv::putText(show, "Rectified check (epipolar lines should match)", {20,40},
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,255,0}, 2);

        cv::imshow("Rectified verify", show);
        std::cout << "Cierra la ventana para terminar.\n";
        cv::waitKey(0);
    }

    return 0;
}
