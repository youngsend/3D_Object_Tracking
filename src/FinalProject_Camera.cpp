
/* INCLUDES FOR THIS PROJECT */
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
#include "matching2D.hpp"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    std::string detectorType = "SHITOMASI"; // SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
    std::string descriptorType = "ORB"; // BRIEF, ORB, FREAK, AKAZE, SIFT
    std::string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
    std::string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

    // specify detector and descriptor types in command line
    if (argc == 3) {
        detectorType = argv[1];
        descriptorType = argv[2];
    }

    /* INIT VARIABLES AND DATA STRUCTURES */
    Matching2D matching2D(detectorType, descriptorType, matcherType, selectorType);

    // data location
    std::string dataPath = "../";

    // camera
    std::string imgBasePath = dataPath + "images/";
    std::string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
    std::string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 18;   // last file index to load
    int imgStepWidth = 1;
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // object detection
    std::string yoloBasePath = dataPath + "dat/yolo/";
    std::string yoloClassesFile = yoloBasePath + "coco.names";
    std::string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    std::string yoloModelWeights = yoloBasePath + "yolov3.weights";

    // Lidar
    std::string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    std::string lidarFileType = ".bin";

    // calibration data for camera and lidar
    cv::Mat P_rect_00(3,4,cv::DataType<double>::type); // 3x4 projection matrix after rectification
    // 3x3 rectifying rotation to make image planes co-planar
    cv::Mat R_rect_00(4,4,cv::DataType<double>::type);
    cv::Mat RT(4,4,cv::DataType<double>::type); // rotation matrix and translation vector

    RT.at<double>(0,0) = 7.533745e-03; RT.at<double>(0,1) = -9.999714e-01;
    RT.at<double>(0,2) = -6.166020e-04; RT.at<double>(0,3) = -4.069766e-03;
    RT.at<double>(1,0) = 1.480249e-02; RT.at<double>(1,1) = 7.280733e-04;
    RT.at<double>(1,2) = -9.998902e-01; RT.at<double>(1,3) = -7.631618e-02;
    RT.at<double>(2,0) = 9.998621e-01; RT.at<double>(2,1) = 7.523790e-03;
    RT.at<double>(2,2) = 1.480755e-02; RT.at<double>(2,3) = -2.717806e-01;
    RT.at<double>(3,0) = 0.0; RT.at<double>(3,1) = 0.0;
    RT.at<double>(3,2) = 0.0; RT.at<double>(3,3) = 1.0;

    R_rect_00.at<double>(0,0) = 9.999239e-01; R_rect_00.at<double>(0,1) = 9.837760e-03;
    R_rect_00.at<double>(0,2) = -7.445048e-03; R_rect_00.at<double>(0,3) = 0.0;
    R_rect_00.at<double>(1,0) = -9.869795e-03; R_rect_00.at<double>(1,1) = 9.999421e-01;
    R_rect_00.at<double>(1,2) = -4.278459e-03; R_rect_00.at<double>(1,3) = 0.0;
    R_rect_00.at<double>(2,0) = 7.402527e-03; R_rect_00.at<double>(2,1) = 4.351614e-03;
    R_rect_00.at<double>(2,2) = 9.999631e-01; R_rect_00.at<double>(2,3) = 0.0;
    R_rect_00.at<double>(3,0) = 0; R_rect_00.at<double>(3,1) = 0;
    R_rect_00.at<double>(3,2) = 0; R_rect_00.at<double>(3,3) = 1;

    P_rect_00.at<double>(0,0) = 7.215377e+02; P_rect_00.at<double>(0,1) = 0.000000e+00;
    P_rect_00.at<double>(0,2) = 6.095593e+02; P_rect_00.at<double>(0,3) = 0.000000e+00;
    P_rect_00.at<double>(1,0) = 0.000000e+00; P_rect_00.at<double>(1,1) = 7.215377e+02;
    P_rect_00.at<double>(1,2) = 1.728540e+02; P_rect_00.at<double>(1,3) = 0.000000e+00;
    P_rect_00.at<double>(2,0) = 0.000000e+00; P_rect_00.at<double>(2,1) = 0.000000e+00;
    P_rect_00.at<double>(2,2) = 1.000000e+00; P_rect_00.at<double>(2,3) = 0.000000e+00;

    // misc
    double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    std::vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex+=imgStepWidth) {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        std::ostringstream imgNumber;
        imgNumber << std::setfill('0') << std::setw(imgFillWidth) << imgStartIndex + imgIndex;
        std::string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file 
        cv::Mat img = cv::imread(imgFullFilename);

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = img;
        uint current_index = imgIndex % dataBufferSize;

        if (dataBuffer.size() < dataBufferSize) {
            // so that I do not need to reserve dataBufferSize space.
            dataBuffer.push_back(frame);
        } else {
            dataBuffer[current_index] = frame;
        }

        std::cout << "#1 : LOAD IMAGE INTO BUFFER done\n";

        /* DETECT & CLASSIFY OBJECTS */

        float confThreshold = 0.2;
        float nmsThreshold = 0.4;
        detectObjects(dataBuffer[current_index].cameraImg, dataBuffer[current_index].boundingBoxes,
                      confThreshold, nmsThreshold,
                      yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);

        std::cout << "#2 : DETECT & CLASSIFY OBJECTS done\n";

        /* CROP LIDAR POINTS */

        // load 3D Lidar points from file
        std::string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
        std::vector<LidarPoint> lidarPoints;
        loadLidarFromFile(lidarPoints, lidarFullFilename);

        // remove Lidar points based on distance properties
        float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane
        cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);

        dataBuffer[current_index].lidarPoints = lidarPoints;

        std::cout << "#3 : CROP LIDAR POINTS done\n";

        /* CLUSTER LIDAR POINT CLOUD */

        // associate Lidar points with camera-based ROI
        // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
        float shrinkFactor = 0.10;
        clusterLidarWithROI(dataBuffer[current_index].boundingBoxes, dataBuffer[current_index].lidarPoints,
                            shrinkFactor, P_rect_00, R_rect_00, RT);

        // Visualize 3D objects
        bVis = true;
        if(bVis)
        {
            show3DObjects(dataBuffer[current_index].boundingBoxes,
                          cv::Size(4.0, 20.0),
                          cv::Size(2000, 2000), true);
        }
        bVis = false;

        std::cout << "#4 : CLUSTER LIDAR POINT CLOUD done\n";

        /* DETECT IMAGE KEYPOINTS */

        // convert current image to grayscale
        cv::Mat imgGray;
        cv::cvtColor(dataBuffer[current_index].cameraImg, imgGray, cv::COLOR_BGR2GRAY);

        // extract 2D keypoints from current image
        std::vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        matching2D.DetectKeypoints(keypoints, imgGray);
//        matching2D.DisplayKeypoints(keypoints, img);

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        // this cropping should be disabled finally when fusing with lidar point cloud.
        matching2D.CropKeypoints(vehicleRect, keypoints);
//        matching2D.DisplayKeypoints(keypoints, img);

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            std::cout << " NOTE: Keypoints have been limited!\n";
        }

        // push keypoints and descriptor for current frame to end of data buffer
        dataBuffer[current_index].keypoints = keypoints;

        /* EXTRACT KEYPOINT DESCRIPTORS */
        cv::Mat descriptors;
        matching2D.DescKeypoints(dataBuffer[current_index].keypoints,
                                 dataBuffer[current_index].cameraImg, descriptors);

        // push descriptors for current frame to end of data buffer
        dataBuffer[current_index].descriptors = descriptors;

        std::cout << "#6 : EXTRACT DESCRIPTORS done\n";

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {
            uint last_index = (imgIndex - 1) % dataBufferSize;
            /* MATCH KEYPOINT DESCRIPTORS */

            std::vector<cv::DMatch> matches;

            matching2D.MatchDescriptors(dataBuffer[last_index].keypoints, dataBuffer[current_index].keypoints,
                                        dataBuffer[last_index].descriptors,
                                        dataBuffer[current_index].descriptors,
                                        matches);

            // store matches in current data frame
            dataBuffer[current_index].kptMatches = matches;

            std::cout << "#7 : MATCH KEYPOINT DESCRIPTORS done\n";

            /* TRACK 3D OBJECT BOUNDING BOXES */

            //// STUDENT ASSIGNMENT
            //// TASK FP.1 -> match list of 3D objects (vector<BoundingBox>) between current and previous frame (implement ->matchBoundingBoxes)
            std::map<int, int> bbBestMatches;
            matchBoundingBoxes(matches, bbBestMatches, dataBuffer[last_index],
                               dataBuffer[current_index]); // associate bounding boxes between current and previous frame using keypoint matches
            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            dataBuffer[current_index].bbMatches = bbBestMatches;

            std::cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done\n";

            /* COMPUTE TTC ON OBJECT IN FRONT */

            // loop over all BB match pairs
            for (auto& bbMatch : dataBuffer[current_index].bbMatches){
                // find bounding boxes associates with current match
                BoundingBox *prevBB, *currBB;
                for (auto& boundingBox : dataBuffer[current_index].boundingBoxes) {
                    if (bbMatch.second == boundingBox.boxID) // check wether current match partner corresponds to this BB
                    {
                        currBB = &(boundingBox);
                    }
                }

                for (auto& boundingBox : dataBuffer[last_index].boundingBoxes){
                    if (bbMatch.first == boundingBox.boxID) // check wether current match partner corresponds to this BB
                    {
                        prevBB = &(boundingBox);
                    }
                }

                // compute TTC for current match
                if(!(currBB->lidarPoints.empty()) && !(prevBB->lidarPoints.empty())){
                    //// STUDENT ASSIGNMENT
                    //// TASK FP.2 -> compute time-to-collision based on Lidar data (implement -> computeTTCLidar)
                    double ttcLidar;
                    computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar);
                    //// EOF STUDENT ASSIGNMENT

                    //// STUDENT ASSIGNMENT
                    //// TASK FP.3 -> assign enclosed keypoint matches to bounding box (implement -> clusterKptMatchesWithROI)
                    //// TASK FP.4 -> compute time-to-collision based on camera (implement -> computeTTCCamera)
                    double ttcCamera;
                    clusterKptMatchesWithROI(*currBB, dataBuffer[last_index].keypoints,
                                             dataBuffer[current_index].keypoints,
                                             dataBuffer[current_index].kptMatches);
                    computeTTCCamera(dataBuffer[last_index].keypoints, dataBuffer[current_index].keypoints,
                                     currBB->kptMatches, sensorFrameRate, ttcCamera);
                    //// EOF STUDENT ASSIGNMENT

                    bVis = true;
                    if (bVis)
                    {
                        cv::Mat visImg = dataBuffer[current_index].cameraImg.clone();
                        showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT,
                                            &visImg);
                        cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y),
                                      cv::Point(currBB->roi.x + currBB->roi.width,
                                                currBB->roi.y + currBB->roi.height),
                                      cv::Scalar(0, 255, 0), 2);

                        char str[200];
                        sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);
                        putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN,
                                2, cv::Scalar(0,0,255));

                        std::string windowName = "Final Results : TTC";
                        cv::namedWindow(windowName, 4);
                        cv::imshow(windowName, visImg);
                        std::cout << "Press key to continue to next frame\n";
                        cv::waitKey(0);
                    }
                    bVis = false;

                } // eof TTC computation
            } // eof loop over all BB matches            

        }

    } // eof loop over all images

    return 0;
}
