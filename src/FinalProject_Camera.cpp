#include "FinalProject_Camera.h"

FinalProjectCamera::FinalProjectCamera() {
    // data location
    dataPath = "../";

    // camera
    imgBasePath = dataPath + "images/";
    imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
    imgFileType = ".png";

    // object detection
    yoloBasePath = dataPath + "dat/yolo/";
    yoloClassesFile = yoloBasePath + "coco.names";
    yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    yoloModelWeights = yoloBasePath + "yolov3.weights";

    // Lidar
    lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    lidarFileType = ".bin";

    // calibration data for camera and lidar
    // 3x4 projection matrix after rectification
    P_rect_00 = cv::Mat(3,4,cv::DataType<double>::type);
    // 3x3 rectifying rotation to make image planes co-planar
    R_rect_00 = cv::Mat(4,4,cv::DataType<double>::type);
    RT = cv::Mat(4,4,cv::DataType<double>::type); // rotation matrix and translation vector

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

    shrinkFactor = 0.10;
    // create CamFusion object using calibration data
    camFusion = std::unique_ptr<CamFusion>(
            new CamFusion(P_rect_00, R_rect_00, RT, shrinkFactor));
}

TTCPairVector FinalProjectCamera::MainProcess(std::unique_ptr<Matching2D> matching2D, bool visTTC) {
    TTCPairVector ttcPairVector;

    // misc
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 18;   // last file index to load
    int imgStepWidth = 1;
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)
    double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    std::vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex+=imgStepWidth) {
        // assemble filenames for current index
        std::ostringstream imgNumber;
        imgNumber << std::setfill('0') << std::setw(imgFillWidth) << imgStartIndex + imgIndex;
        std::string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file
        cv::Mat img = cv::imread(imgFullFilename);

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = img;
        uint curr_index = imgIndex % dataBufferSize;

        if (dataBuffer.size() < dataBufferSize) {
            // so that I do not need to reserve dataBufferSize space.
            dataBuffer.push_back(frame);
        } else {
            dataBuffer[curr_index] = frame;
        }

        float confThreshold = 0.2;
        float nmsThreshold = 0.4;
        detectObjects(dataBuffer[curr_index].cameraImg, dataBuffer[curr_index].boundingBoxes,
                      confThreshold, nmsThreshold,
                      yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);

        // load 3D Lidar points from file
        std::string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
        std::vector<LidarPoint> lidarPoints;
        loadLidarFromFile(lidarPoints, lidarFullFilename);

        // remove Lidar points based on distance properties
        float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane
        cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);

        dataBuffer[curr_index].lidarPoints = lidarPoints;

        // associate Lidar points with camera-based ROI
        // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
        camFusion->ClusterLidarWithROI(dataBuffer[curr_index].boundingBoxes,
                                       dataBuffer[curr_index].lidarPoints);

        // Visualize 3D objects
//        camFusion.Display3DObjects(dataBuffer[curr_index].boundingBoxes,
//                                   cv::Size(4.0, 20.0),
//                                   cv::Size(2000, 2000));

        // convert current image to grayscale
        cv::Mat imgGray;
        cv::cvtColor(dataBuffer[curr_index].cameraImg, imgGray, cv::COLOR_BGR2GRAY);

        // extract 2D keypoints from current image
        matching2D->DetectKeypoints(dataBuffer[curr_index].keypoints, imgGray);

        matching2D->DescKeypoints(dataBuffer[curr_index].keypoints,
                                  dataBuffer[curr_index].cameraImg, dataBuffer[curr_index].descriptors);

        // wait until at least two images have been processed
        if (dataBuffer.size() > 1){
            uint prev_index = (imgIndex - 1) % dataBufferSize;

            matching2D->MatchDescriptors(dataBuffer[prev_index].keypoints,
                                         dataBuffer[curr_index].keypoints,
                                         dataBuffer[prev_index].descriptors,
                                         dataBuffer[curr_index].descriptors,
                                         dataBuffer[curr_index].kptMatches);

            std::cout << "matches size: " << dataBuffer[curr_index].kptMatches.size() << "\n";

            // associate bounding boxes between current and previous frame using keypoint matches
            camFusion->MatchBoundingBoxes(dataBuffer[curr_index].kptMatches,
                                          dataBuffer[curr_index].bbMatches,
                                          dataBuffer[prev_index].boundingBoxes,
                                          dataBuffer[curr_index].boundingBoxes,
                                          dataBuffer[prev_index].keypoints,
                                          dataBuffer[curr_index].keypoints);

//            camFusion->DisplayTwoFramesWithBoundingBoxMatch(dataBuffer[prev_index],
//                                                            dataBuffer[curr_index]);

            // loop over all BB match pairs.
            for (auto& bbMatch : dataBuffer[curr_index].bbMatches){
                // find bounding boxes associates with current match. From detectObject, boxID is its index.
                assert(bbMatch.first < dataBuffer[prev_index].boundingBoxes.size());
                assert(bbMatch.second < dataBuffer[curr_index].boundingBoxes.size());
                auto& prevBB = dataBuffer[prev_index].boundingBoxes.at(bbMatch.first);
                auto& currBB = dataBuffer[curr_index].boundingBoxes.at(bbMatch.second);

                // compute TTC for current match
                if(!(currBB.lidarPoints.empty()) && !(prevBB.lidarPoints.empty())){
                    //// TASK FP.2 -> compute time-to-collision based on Lidar data
                    double ttcLidar = camFusion->ComputeTTCLidar(prevBB.lidarPoints, currBB.lidarPoints,
                                                                 sensorFrameRate);

                    //// TASK FP.3 -> assign enclosed keypoint matches to bounding box
                    //// TASK FP.4 -> compute time-to-collision based on fusion
                    camFusion->RemoveMatchOutliersRansac(dataBuffer[prev_index].keypoints,
                                                         dataBuffer[curr_index].keypoints,
                                                         currBB.kptMatches);

                    double ttcCamera = camFusion->ComputeTTCCamera(dataBuffer[prev_index].keypoints,
                                                                   dataBuffer[curr_index].keypoints,
                                                                   currBB.kptMatches, sensorFrameRate);

                    // save ttc pair
                    ttcPairVector.emplace_back(ttcLidar, ttcCamera);

                    if (visTTC) {
                        camFusion->DisplayTTC(dataBuffer[curr_index].cameraImg, currBB, ttcLidar, ttcCamera);
                    }
                } // eof TTC computation
            } // eof loop over all BB matches
        }
    } // eof loop over all images
    return ttcPairVector;
}
