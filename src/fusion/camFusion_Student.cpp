
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <set>

#include "camFusion.hpp"

CamFusion::CamFusion(cv::Mat P_rect_xx, cv::Mat R_rect_xx, cv::Mat RT) :
        P_rect_xx(std::move(P_rect_xx)), R_rect_xx(std::move(R_rect_xx)), RT(std::move(RT)){}

// Create groups of Lidar points whose projection into the fusion falls into the same bounding box
void CamFusion::ClusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints,
                                    float shrinkFactor){
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto& lidarPoint : lidarPoints) {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = lidarPoint.x;
        X.at<double>(1, 0) = lidarPoint.y;
        X.at<double>(2, 0) = lidarPoint.z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into fusion
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        // pointers to all bounding boxes which enclose the current Lidar point
        std::vector<BoundingBox*> enclosingBoxes;
        for (auto& boundingBox : boundingBoxes) {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = boundingBox.roi.x + shrinkFactor * boundingBox.roi.width / 2.0;
            smallerBox.y = boundingBox.roi.y + shrinkFactor * boundingBox.roi.height / 2.0;
            smallerBox.width = boundingBox.roi.width * (1 - shrinkFactor);
            smallerBox.height = boundingBox.roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(&boundingBox);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(lidarPoint);
        }

    } // eof loop over all Lidar points
}


void CamFusion::Display3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize){
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto& boundingBox : boundingBoxes) {
        // create randomized color for current 3D object
        cv::RNG rng(boundingBox.boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150),
                                          rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0;
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto& lidarPoint : boundingBox.lidarPoints) {
            // world coordinates
            float xw = lidarPoint.x; // world position in m with x facing forward from sensor
            float yw = lidarPoint.y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),
                      cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", boundingBox.boxID, (int)boundingBox.lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50),
                cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125),
                cv::FONT_ITALIC, 2, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i) {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y),
                 cv::Scalar(255, 0, 0));
    }

    // display image
    std::string windowName = "3D Objects";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::imshow(windowName, topviewImg);
    cv::waitKey(0); // wait for key to be pressed
}


// associate a given bounding box with the keypoints it contains
void CamFusion::ClusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev,
                                         std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches){
    // ...
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
double CamFusion::ComputeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                                   std::vector<cv::DMatch> kptMatches, double frameRate, cv::Mat *visImg){
    double TTC = 0.0;

    return TTC;
}

double CamFusion::ComputeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                                  std::vector<LidarPoint> &lidarPointsCurr, double frameRate){
    float clusterTolerance = 0.05;

    auto prevCluster = GetLargestEuclideanCluster(lidarPointsPrev, clusterTolerance);
    auto d0 = GetClosestLidarPoint(lidarPointsPrev, prevCluster).x;

    auto currCluster = GetLargestEuclideanCluster(lidarPointsCurr, clusterTolerance);
    auto d1 = GetClosestLidarPoint(lidarPointsCurr, currCluster).x;

    double TTC = std::fabs(d1 / (d0-d1) / frameRate);
    std::cout << "d0: " << d0 << " ,d1: " << d1 << std::endl;

    return TTC;
}


void CamFusion::MatchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches,
                                   DataFrame &prevFrame, DataFrame &currFrame) {
    // map<pair<prev frame box id, current frame box id>, DMatch number>
    std::map<std::pair<int, int>, uint32_t> boundingBoxMatchCounts;

    // count the match numbers for <prevBoxId, currBoxId> pairs.
    for(auto& match : matches) {
        auto& prevKeypoint = prevFrame.keypoints[match.trainIdx];
        auto& currKeypoint = currFrame.keypoints[match.queryIdx];
        for(auto& preBoundingBox : prevFrame.boundingBoxes){
            for(auto& currBoundingBox : currFrame.boundingBoxes) {
                // Use this loop because several bounding boxes may overlap, so they may contain the same keypoint.
                if (preBoundingBox.roi.contains(prevKeypoint.pt) &&
                    currBoundingBox.roi.contains(currKeypoint.pt)){
                    // value of map is initialized to 0.
                    boundingBoxMatchCounts[std::make_pair(preBoundingBox.boxID,
                                                          currBoundingBox.boxID)]++;
                }
            }
        }
    }

    // sort the <prevBoxId, currBoxId> by count number in descending order.
    std::vector<BoundingBoxMatchCount> boundingBoxMatchCountVector;
    for(auto& pair : boundingBoxMatchCounts){
        boundingBoxMatchCountVector.emplace_back(pair.first.first, pair.first.second, pair.second);
    }
    std::sort(boundingBoxMatchCountVector.begin(),
              boundingBoxMatchCountVector.end(),
              [](BoundingBoxMatchCount& a, BoundingBoxMatchCount& b){return a.count > b.count;});

    // Take the box match with the largest count number. In bbBestMatches, box id should not duplicate.
    std::set<int> prevBoxIdUsed, currBoxIdUsed;
    for(auto& element : boundingBoxMatchCountVector){
        // when match count is less than or equals 1, do not consider this bounding box match.
        if(element.count <= 1)
            break;
        if(prevBoxIdUsed.insert(element.prevBoxId).second &&
           currBoxIdUsed.insert(element.currBoxId).second){
            // both box ids have not been matched.
            bbBestMatches[element.prevBoxId] = element.currBoxId;
        }
    }
}

void CamFusion::DisplayTTC(const cv::Mat &cameraImg, BoundingBox *currBB, double ttcLidar, double ttcCamera) {
    cv::Mat visImg = cameraImg.clone();
    DisplayLidarImgOverlay(visImg, currBB->lidarPoints, &visImg);
    cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y),
                  cv::Point(currBB->roi.x + currBB->roi.width,
                            currBB->roi.y + currBB->roi.height),
                  cv::Scalar(0, 255, 0), 2);

    char str[200];
    sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);
    putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN,
            2, cv::Scalar(0,0,255));

    std::string windowName = "Final Results : TTC";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    cv::imshow(windowName, visImg);
    std::cout << "Press key to continue to next frame\n";
    cv::waitKey(0);
}

void CamFusion::DisplayLidarImgOverlay(cv::Mat &img, std::vector<LidarPoint> &lidarPoints, cv::Mat *extVisImg){
    // init image for visualization
    cv::Mat visImg;
    if(extVisImg==nullptr)
    {
        visImg = img.clone();
    } else
    {
        visImg = *extVisImg;
    }

    cv::Mat overlay = visImg.clone();

    // find max. x-value
    double maxVal = 0.0;
    for(auto& lidarPoint : lidarPoints){
        maxVal = maxVal < lidarPoint.x ? lidarPoint.x : maxVal;
    }

    cv::Mat X(4,1,cv::DataType<double>::type);
    cv::Mat Y(3,1,cv::DataType<double>::type);
    for(auto& lidarPoint : lidarPoints) {
        X.at<double>(0, 0) = lidarPoint.x;
        X.at<double>(1, 0) = lidarPoint.y;
        X.at<double>(2, 0) = lidarPoint.z;
        X.at<double>(3, 0) = 1;

        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        float val = lidarPoint.x;
        int red = std::min(255, (int)(255 * std::abs((val - maxVal) / maxVal)));
        int green = std::min(255, (int)(255 * (1 - std::abs((val - maxVal) / maxVal))));
        cv::circle(overlay, pt, 5, cv::Scalar(0, green, red), -1);
    }

    float opacity = 0.6;
    cv::addWeighted(overlay, opacity, visImg, 1 - opacity, 0, visImg);

    // return augmented image or wait if no image has been provided
    if (extVisImg == nullptr)
    {
        std::string windowName = "LiDAR data on image overlay";
        cv::namedWindow( windowName, 3 );
        cv::imshow( windowName, visImg );
        cv::waitKey(0); // wait for key to be pressed
    }
    else
    {
        extVisImg = &visImg;
    }
}

LidarPoint CamFusion::GetClosestLidarPoint(const std::vector<LidarPoint> &lidarPoints,
                                           const std::vector<int> &cluster) {
    LidarPoint lidarPoint = lidarPoints.at(0);
    for(auto index : cluster){
        if (lidarPoints.at(index).x < lidarPoint.x){
            lidarPoint = lidarPoints.at(index);
        }
    }
    return lidarPoint;
}