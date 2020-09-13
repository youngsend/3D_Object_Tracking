//
// Created by sen on 2020/09/12.
//

#ifndef CAMERA_FUSION_FINALPROJECT_CAMERA_H
#define CAMERA_FUSION_FINALPROJECT_CAMERA_H

#endif //CAMERA_FUSION_FINALPROJECT_CAMERA_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "camera/matching2D.hpp"
#include "camera/objectDetection2D.hpp"
#include "lidar/lidarData.hpp"
#include "fusion/camFusion.hpp"

class FinalProjectCamera {
public:
    FinalProjectCamera();
    ~FinalProjectCamera() = default;
    void MainProcess(Matching2D& matching2D);

private:
    // data location
    std::string dataPath;

    // camera
    std::string imgBasePath;
    std::string imgPrefix; // left camera, color
    std::string imgFileType;

    // object detection
    std::string yoloBasePath;
    std::string yoloClassesFile;
    std::string yoloModelConfiguration;
    std::string yoloModelWeights;

    // Lidar
    std::string lidarPrefix;
    std::string lidarFileType;

    // calibration data for camera and lidar
    cv::Mat P_rect_00; // 3x4 projection matrix after rectification
    // 3x3 rectifying rotation to make image planes co-planar
    cv::Mat R_rect_00;
    cv::Mat RT; // rotation matrix and translation vector

    float shrinkFactor;
    CamFusion* camFusion;
};