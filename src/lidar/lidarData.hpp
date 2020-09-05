
#ifndef lidarData_hpp
#define lidarData_hpp

#include <cstdio>
#include <fstream>
#include <string>
#include <unordered_set>

#include "../dataStructures.h"
#include "kdtree.h"

void cropLidarPoints(std::vector<LidarPoint> &lidarPoints, float minX, float maxX, float maxY, float minZ,
                     float maxZ, float minR);
void loadLidarFromFile(std::vector<LidarPoint> &lidarPoints, std::string filename);

void showLidarTopview(std::vector<LidarPoint> &lidarPoints, cv::Size worldSize, cv::Size imageSize, bool bWait=true);

void Proximity(std::unordered_set<int>& processed_ids, const std::vector<LidarPoint>& cloud,
               std::vector<int>& cluster_ids, int index, KdTree* tree, float distanceTol);

// cluster and only return the largest cluster
std::vector<int> GetLargestEuclideanCluster(const std::vector<LidarPoint>& cloud,
                                            float distanceTol);

#endif /* lidarData_hpp */
