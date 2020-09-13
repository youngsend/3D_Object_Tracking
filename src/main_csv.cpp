//
// Created by sen on 2020/09/13.
//

#include "FinalProject_Camera.h"

void OutputLidarTTC(std::ofstream& of, const TTCPairVector& ttcPairVector,
                    std::string detectorType, std::string descriptorType){
    of << "\n" << detectorType+"_"+descriptorType;
    for(auto& pair : ttcPairVector){
        of << "," << pair.first;
    }
}

void OutputCameraTTC(std::ofstream& of, const TTCPairVector& ttcPairVector,
                     std::string detectorType, std::string descriptorType){
    of << "\n" << detectorType+"_"+descriptorType;
    for(auto& pair : ttcPairVector){
        of << "," << pair.second;
    }
}

int main(int argc, const char *argv[]){
    std::string csv_path = "../";
    std::string csv_name = "ttc.csv";
    std::ofstream outputCSV;

    FinalProjectCamera finalProjectCamera;

    std::vector<std::string> detectorTypes =
            {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB"};
    std::vector<std::string> descriptorTypes = {"BRIEF", "ORB", "FREAK"};
    std::string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
    std::string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

    outputCSV.open(csv_path + csv_name);
    auto ttcAkaze = finalProjectCamera.MainProcess(std::unique_ptr<Matching2D>(
            new Matching2D("AKAZE", "AKAZE",
                           matcherType, selectorType)), false);
    OutputLidarTTC(outputCSV, ttcAkaze, "AKAZE", "AKAZE");
    OutputCameraTTC(outputCSV, ttcAkaze, "AKAZE", "AKAZE");

    auto ttcSift = finalProjectCamera.MainProcess(std::unique_ptr<Matching2D>(
            new Matching2D("SIFT", "SIFT",
                           matcherType, selectorType)), false);
    OutputCameraTTC(outputCSV, ttcSift, "SIFT", "SIFT");

    for(auto detectorType : detectorTypes) {
        for(auto descriptorType : descriptorTypes) {
            auto ttcPairs = finalProjectCamera.MainProcess(std::unique_ptr<Matching2D>(
                    new Matching2D(detectorType, descriptorType, matcherType, selectorType)), false);
            OutputCameraTTC(outputCSV, ttcPairs, detectorType, descriptorType);
        }
    }

    outputCSV.close();
    return 0;
}
