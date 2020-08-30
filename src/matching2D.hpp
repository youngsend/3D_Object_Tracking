#ifndef matching2D_hpp
#define matching2D_hpp

#include <cstdio>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"

class Matching2D {
public:
    explicit Matching2D(std::string detectorType = "ORB",
                        std::string descriptorType = "ORB",
                        std::string matcherType = "MAT_BF",
                        std::string selectorType = "SEL_KNN");
    ~Matching2D();

    // keypoint detector
    void DetectKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img);

    // keypoint descriptor
    void DescKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors);
    void MatchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef,
                          cv::Mat &descSource, cv::Mat &descRef,
                          std::vector<cv::DMatch> &matches);

    // helper functions
    void DisplayKeypoints(std::vector<cv::KeyPoint>& keypoints, cv::Mat& img);
    void DisplayMatches(const DataFrame& current, const DataFrame& last, const std::vector<cv::DMatch>& matches);
    void CropKeypoints(const cv::Rect& rect, std::vector<cv::KeyPoint>& keypoints);

    // calculate mean and std for keypoints' neighborhood size
    void CalculateNeighborhoodDistribution(const std::vector<cv::KeyPoint> &keypoints);

private:
    void DetKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img);
    void DetKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img);
    void DetKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img);

private:
    std::ofstream outputCSV;
    std::string detectorType;
    std::string descriptorType;
    std::string matcherType;
    std::string selectorType;
    std::string descriptorBigType;
};

#endif /* matching2D_hpp */
