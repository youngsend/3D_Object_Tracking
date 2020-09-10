#include <iostream>
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "lidarData.hpp"

// remove Lidar points based on min. and max distance in X, Y and Z
void cropLidarPoints(std::vector<LidarPoint> &lidarPoints, float minX, float maxX, float maxY, float minZ,
                     float maxZ, float minR){
    std::vector<LidarPoint> newLidarPts;
    for(auto& lidarPoint : lidarPoints) {
        if(lidarPoint.x >= minX && lidarPoint.x <= maxX && lidarPoint.z >= minZ &&
           lidarPoint.z <= maxZ && lidarPoint.z <= 0.0 &&
           abs(lidarPoint.y) <= maxY && lidarPoint.r >= minR )  // Check if Lidar point is outside of boundaries
        {
            newLidarPts.push_back(lidarPoint);
        }
    }

    lidarPoints = newLidarPts;
}

// Load Lidar points from a given location and store them in a vector
void loadLidarFromFile(std::vector<LidarPoint> &lidarPoints, std::string filename){
    // allocate 4 MB buffer (only ~130*4*4 KB are needed)
    unsigned long num = 1000000;
    auto *data = (float*)malloc(num*sizeof(float));

    // pointers
    float *px = data+0;
    float *py = data+1;
    float *pz = data+2;
    float *pr = data+3;

    // load point cloud
    FILE *stream;
    stream = fopen (filename.c_str(),"rb");
    num = fread(data,sizeof(float),num,stream)/4;

    for (int32_t i=0; i<num; i++) {
        LidarPoint lpt;
        lpt.x = *px; lpt.y = *py; lpt.z = *pz; lpt.r = *pr;
        lidarPoints.push_back(lpt);
        px+=4; py+=4; pz+=4; pr+=4;
    }
    fclose(stream);
}

void showLidarTopview(std::vector<LidarPoint> &lidarPoints, cv::Size worldSize,
                      cv::Size imageSize, bool bWait){
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(0, 0, 0));

    // plot Lidar points into image
    for (auto& lidarPoint : lidarPoints){
        float xw = lidarPoint.x; // world position in m with x facing forward from sensor
        float yw = lidarPoint.y; // world position in m with y facing left from sensor

        int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
        int x = (-yw * imageSize.height / worldSize.height) + imageSize.width / 2;

        cv::circle(topviewImg, cv::Point(x, y), 5, cv::Scalar(0, 0, 255), -1);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y),
                 cv::Scalar(255, 0, 0));
    }

    // display image
    std::string windowName = "Top-View Perspective of LiDAR data";
    cv::namedWindow(windowName, 2);
    cv::imshow(windowName, topviewImg);
    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

void Proximity(std::unordered_set<int> &processed_ids,
               const std::vector<LidarPoint>& cloud,
               std::vector<int> &cluster_ids,
               int index, KdTree *tree, float distanceTol) {
    processed_ids.insert(index);
    cluster_ids.push_back(index);
    auto nearby_points = tree->search(cloud[index], distanceTol);
    for (int nearby_index : nearby_points) {
        if (!processed_ids.count(nearby_index)) {
            Proximity(processed_ids, cloud, cluster_ids, nearby_index, tree, distanceTol);
        }
    }
}

std::vector<int> GetLargestEuclideanCluster(const std::vector<LidarPoint>& cloud,
                                            float distanceTol){
    // Build kd-tree
    KdTree* tree = new KdTree();
    for(int i=0; i<cloud.size(); i++){
        tree->insert(cloud[i], i);
    }

    // Fill out this function to return list of indices for each cluster
    std::vector<int> largest_cluster;
    std::unordered_set<int> processed_ids;
    for (int index=0; index<cloud.size(); index++){
        if(!processed_ids.count(index)) {
            std::vector<int> cluster_ids;
            Proximity(processed_ids, cloud, cluster_ids, index, tree, distanceTol);
            if(cluster_ids.size() > largest_cluster.size()) {
                largest_cluster = cluster_ids;
            }
        }
    }
//    std::cout << "largest cluster size: " << largest_cluster.size() << "\n";
    return largest_cluster;
}