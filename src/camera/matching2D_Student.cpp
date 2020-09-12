#include <numeric>
#include "matching2D.hpp"

#include <vector>
#include <string>

Matching2D::Matching2D(std::string detectorType, std::string descriptorType, std::string matcherType,
                       std::string selectorType) :
        detectorType(std::move(detectorType)),
        descriptorType(std::move(descriptorType)),
        matcherType(std::move(matcherType)),
        selectorType(std::move(selectorType)){
    // decide whether the descriptor is binary or hog, needed in MatchDescriptors.
    if (descriptorType == "BRISK" ||
        descriptorType == "BRIEF" ||
        descriptorType == "ORB" ||
        descriptorType == "FREAK" ||
        descriptorType == "AKAZE") {
        descriptorBigType = "DES_BINARY";
    } else {
        descriptorBigType = "DES_HOG";
    }
}

// Find best matches for keypoints in two fusion images based on several matching methods
void Matching2D::MatchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef,
                                  cv::Mat &descSource, cv::Mat &descRef,
                                  std::vector<cv::DMatch> &matches) {
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    int normType = cv::NORM_HAMMING;
    if (descriptorBigType == "DES_HOG") {
        normType = cv::NORM_L2;
    }

    if (matcherType == "MAT_BF") {
        matcher = cv::BFMatcher::create(normType, crossCheck);
    } else if (matcherType == "MAT_FLANN") {
        if (descriptorBigType == "DES_HOG") {
            matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        } else if (descriptorBigType == "DES_BINARY") {
            matcher = cv::makePtr<cv::FlannBasedMatcher>(
                    cv::makePtr<cv::flann::LshIndexParams>(12,20, 2));
        }
    }

    // perform matching task
    if (selectorType == "SEL_NN") { // nearest neighbor (best match)
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    } else if (selectorType == "SEL_KNN") { // k nearest neighbors (k=2)
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, 2);

        // Filter matches using the Lowe's ratio test
        const float ratio_thresh = 0.8f;
        for (auto& kMatches : knn_matches) {
            if (kMatches.size() >= 2 && kMatches[0].distance < ratio_thresh * kMatches[1].distance) {
                matches.push_back(kMatches[0]);
            }
        }
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void Matching2D::DescKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors) {
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType =="BRISK") {
        int threshold = 30;
        int octaves = 3;
        float patternScale = 1.0f;
        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    } else if(descriptorType == "BRIEF"){
        int bytes = 32;
        bool use_orientation = false;
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, use_orientation);
    } else if(descriptorType == "ORB") {
        int nfeatures = 500;
        float scaleFactor = 1.2f;
        int nlevels = 8;
        int edgeThreshold = 31;
        int firstLevel = 0;
        int WTA_K = 2;
        cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;
        int patchSize = 31;
        int fastThreshold = 20;
        extractor = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold,
                                    firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
    } else if(descriptorType == "FREAK") {
        bool orientationNormalized = true;
        bool scaleNormalized = true;
        float patternScale = 22.0f;
        int nOctaves = 4;
        extractor = cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized, patternScale, nOctaves);
    } else if (descriptorType == "AKAZE") {
        cv::AKAZE::DescriptorType descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
        int descriptor_size = 0;
        int descriptor_channels = 3;
        float threshold = 0.001f;
        int nOctaves = 4;
        int nOctaveLayers = 4;
        cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2;
        extractor = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels,
                                      threshold, nOctaves, nOctaveLayers, diffusivity);
    } else if (descriptorType == "SIFT") {
        int nfeatures = 0;
        int nOctaveLayers = 3;
        double contrastThreshold = 0.04;
        double edgeThreshold = 10;
        double sigma = 1.6;
        extractor = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold,
                                                  edgeThreshold, sigma);
    } else {
        std::cerr << "Error: This extractor type does not exist or has not been implemented!\n";
        return;
    }

    // perform feature description
    double t = cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << std::endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void Matching2D::DetKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img)
{
    // compute detector parameters based on image size
    //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    int blockSize = 4;
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / std::max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    std::vector<cv::Point2f> corners;
    double t = cv::getTickCount();
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize,
                            false, k);

    // add corners to result vector
    for (const auto& corner : corners) {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f(corner.x, corner.y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in "
              << 1000 * t / 1.0 << " ms\n";
}

// https://docs.opencv.org/4.1.0/d4/d7d/tutorial_harris_detector.html
void Matching2D::DetKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img) {
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    double t = cv::getTickCount();
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // Look for prominent corners and instantiate keypoints
    for (size_t j = 0; j < dst_norm.rows; j++) {
        for (size_t i = 0; i < dst_norm.cols; i++) {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse) { // only store points above a threshold
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;
                keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
            }
        } // eof loop over cols
    }     // eof loop over rows
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "Harris corner detection with n=" << keypoints.size() << " keypoints in "
              << 1000 * t / 1.0 << " ms\n";
}

// Refer to https://docs.opencv.org/4.1.0/d0/d13/classcv_1_1Feature2D.html
void Matching2D::DetKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img){
    // cv::FeatureDetector and cv::DescriptorExtractor are the same. Both are aliases of cv::Feature2D.
    cv::Ptr<cv::FeatureDetector> detector;
    if (detectorType == "FAST") {
        // difference between intensity of the central pixel and pixels of a circle around this pixel
        int threshold = 30;
        bool bNMS = true; // perform non-maxima suppression on keypoints
        // TYPE_9_16, TYPE_7_12, TYPE_5_8
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;
        detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
    } else if (detectorType == "BRISK") {
        int thresh = 30; // AGAST detection threshold score.
        int octaves = 3; // detection octaves.
        float patternScale = 1.0f;
        detector = cv::BRISK::create(thresh, octaves, patternScale);
    } else if (detectorType == "ORB") {
        int nfeatures = 500; // The maximum number of features to retain.
        float scaleFactor = 1.2f; // Pyramid decimation ratio, greater than 1.
        int nlevels = 8; // the number of pyramid levels.
        int edgeThreshold = 31; // Size of the border where the features are not detected.
        int firstLevel = 0; // The level of pyramid to put source image to.
        int WTA_K = 2; // The number of points that produce each element of the oriented BRIEF descriptor.
        cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;
        int patchSize = 31; // Size of patch used by the oriented BRIEF descriptor.
        int fastThreshold = 20;
        detector = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel,
                                   WTA_K, scoreType, patchSize, fastThreshold);
    } else if (detectorType == "AKAZE") {
        cv::AKAZE::DescriptorType descriptorType = cv::AKAZE::DESCRIPTOR_MLDB;
        int descriptorSize = 0; // Size of the descriptor in bits. 0->Full.
        int descriptorChannels = 3; // Number of channels in the descriptor (1,2,3).
        float threshold = 0.001f; // Detector response threshold to accept point.
        int nOctaves = 4; // Maximum octave evolution of the image.
        int nOctaveLayers = 4; // Default number of sublevels per scale level.
        cv::KAZE::DiffusivityType diffusivityType = cv::KAZE::DIFF_PM_G2;
        detector = cv::AKAZE::create(descriptorType, descriptorSize, descriptorChannels,
                                     threshold, nOctaves, nOctaveLayers, diffusivityType);
    } else if (detectorType == "SIFT") {
        int nfeatures = 0;
        int nOctaveLayers = 3;
        double contrastThreshold = 0.04;
        double edgeThreshold = 10;
        double sigma = 1.6;
        detector = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold,
                                                 edgeThreshold, sigma);
    } else {
        std::cerr << "Error: This detector type does not exist or has not been implemented!\n";
        return;
    }

    double t = cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << detectorType << " with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms\n";
}

void Matching2D::DisplayKeypoints(std::vector<cv::KeyPoint>& keypoints, cv::Mat& img){
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    std::string windowName = detectorType + " Detection Results";
    cv::namedWindow(windowName, 5);
    imshow(windowName, visImage);
    cv::waitKey(0);
}

void
Matching2D::DisplayMatches(const DataFrame &current, const DataFrame &last, const std::vector<cv::DMatch> &matches) {
    cv::Mat matchImg = (current.cameraImg).clone();
    cv::drawMatches(last.cameraImg, last.keypoints,
                    current.cameraImg, current.keypoints,
                    matches, matchImg,
                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    std::string windowName = "Matching keypoints between two camera images";
    cv::namedWindow(windowName, 7);
    cv::imshow(windowName, matchImg);
    cv::waitKey(0); // wait for key to be pressed
}

void Matching2D::CropKeypoints(const cv::Rect &rect, std::vector<cv::KeyPoint> &keypoints) {
    // remove the keypoint from keypoints if rect does not contain it.
    keypoints.erase(std::remove_if(keypoints.begin(), keypoints.end(),
                                   [&](const cv::KeyPoint& keyPoint){return !rect.contains(keyPoint.pt);}),
                    keypoints.end());
}

void Matching2D::DetectKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img) {
    if (detectorType == "SHITOMASI") {
        DetKeypointsShiTomasi(keypoints, img);
    } else if (detectorType == "HARRIS") {
        DetKeypointsHarris(keypoints, img);
    } else {
        DetKeypointsModern(keypoints, img);
    }
}
