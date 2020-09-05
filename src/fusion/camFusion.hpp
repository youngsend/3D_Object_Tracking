
#ifndef camFusion_hpp
#define camFusion_hpp

#include <cstdio>
#include <vector>
#include <opencv2/core.hpp>
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
    explicit CamFusion(cv::Mat P_rect_xx, cv::Mat R_rect_xx, cv::Mat RT);
    // fusion
    void ClusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints,
                             float shrinkFactor);
    void ClusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev,
                                  std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches);
    void MatchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches,
                            DataFrame &prevFrame, DataFrame &currFrame);

    // display
    void Display3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize);
    void DisplayTTC(const cv::Mat& cameraImg, BoundingBox* currBB, double ttcLidar, double ttcCamera);
    void DisplayLidarImgOverlay(cv::Mat &img, std::vector<LidarPoint> &lidarPoints, cv::Mat *extVisImg= nullptr);

    // compute TTC
    double ComputeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                            std::vector<cv::DMatch> kptMatches, double frameRate, cv::Mat *visImg = nullptr);
    double ComputeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                           std::vector<LidarPoint> &lidarPointsCurr, double frameRate);

    // helper
    LidarPoint GetClosestLidarPoint(const std::vector<LidarPoint>& lidarPoints,
                                    const std::vector<int>& cluster);

private:
    cv::Mat P_rect_xx;
    cv::Mat R_rect_xx;
    cv::Mat RT;
};
#endif /* camFusion_hpp */
