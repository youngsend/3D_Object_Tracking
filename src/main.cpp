//
// Created by sen on 2020/09/13.
//

#include "FinalProject_Camera.h"

int main(int argc, const char *argv[]){
    std::string detectorType = "SHITOMASI"; // SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
    std::string descriptorType = "ORB"; // BRIEF, ORB, FREAK, AKAZE, SIFT
    std::string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
    std::string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

    // specify detector and descriptor types in command line
    if (argc == 3) {
        detectorType = argv[1];
        descriptorType = argv[2];
    }

    FinalProjectCamera finalProjectCamera;
    Matching2D matching2D(detectorType, descriptorType, matcherType, selectorType);
    finalProjectCamera.MainProcess(matching2D);

    return 0;
}
