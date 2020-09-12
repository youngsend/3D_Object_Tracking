
#ifndef camFusion_hpp
#define camFusion_hpp

#include <cstdio>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include "../dataStructures.h"
#include "../lidar/lidarData.hpp"

struct BoundingBoxMatchCount {
    int prevBoxId;
    int currBoxId;
    uint32_t count;
    BoundingBoxMatchCount(int prevBoxId, int currBoxId, uint32_t count):
            prevBoxId(prevBoxId), currBoxId(currBoxId), count(count){}
};

class CamFusion {
public:
    explicit CamFusion(cv::Mat P_rect_xx, cv::Mat R_rect_xx, cv::Mat RT, float shrinkFactor);
    // fusion
    void ClusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints);
    void MatchBoundingBoxes(const std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches,
                            const std::vector<BoundingBox>& prevBoxes,
                            std::vector<BoundingBox>& currBoxes,
                            const std::vector<cv::KeyPoint>& prevKeypoints,
                            const std::vector<cv::KeyPoint>& currKeypoints);

    // display
    void Display3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize);
    void DisplayTTC(const cv::Mat& cameraImg, BoundingBox& currBB, double ttcLidar, double ttcCamera);
    void DisplayLidarImgOverlay(cv::Mat &img, std::vector<LidarPoint> &lidarPoints, cv::Mat *extVisImg= nullptr);
    void AddBoundingBoxesToImg(cv::Mat& img, std::vector<BoundingBox>& boundingBoxes);
    void DisplayWindow(const std::string& windowName, cv::Mat& visImg);
    void DisplayTwoImages(cv::Mat& imgPrev, cv::Mat& imgCurr);
    void DisplayTwoFramesWithBoundingBoxMatch(DataFrame& prevFrame, DataFrame& currFrame);

    // compute TTC
    double ComputeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                            std::vector<cv::DMatch>& kptMatches, double frameRate, cv::Mat *visImg = nullptr);
    double ComputeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                           std::vector<LidarPoint> &lidarPointsCurr, double frameRate);

    // helper
    LidarPoint GetClosestLidarPoint(const std::vector<LidarPoint>& lidarPoints,
                                    const std::vector<int>& cluster);
    void RemoveMatchOutliersRansac(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                                   std::vector<cv::DMatch>& kptMatches);

private:
    cv::Mat P_rect_xx;
    cv::Mat R_rect_xx;
    cv::Mat RT;
    float shrinkFactor;
};
#endif /* camFusion_hpp */
