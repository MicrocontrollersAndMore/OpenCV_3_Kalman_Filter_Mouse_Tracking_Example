// kalman_from_opencvexamples.c

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"

#include<iostream>

// function prototypes ////////////////////////////////////////////////////////////////////////////
void mouseMoveCallback(int event, int x, int y, int flags, void* userData);
void drawCross(cv::Mat &img, cv::Point center, cv::Scalar color);

// global variables ///////////////////////////////////////////////////////////////////////////////
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_BLUE = cv::Scalar(255.0, 0.0, 0.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);
const cv::Scalar SCALAR_TEAL = cv::Scalar(255.0, 255.0, 0.0);
const cv::Scalar SCALAR_BROWN = cv::Scalar(0.0, 155.0, 255.0);

cv::Point ptActualMousePosition(0, 0);

///////////////////////////////////////////////////////////////////////////////////////////////////
int main(void) {

    cv::KalmanFilter kalmanFilter(4, 2, 0);

    float fltTransitionMatrixValues[4][4] = { { 1, 0, 1, 0 },                   // research this !!!!!!!!!!!!!!!!!!!
                                              { 0, 1, 0, 1 },
                                              { 0, 0, 1, 0 },
                                              { 0, 0, 0, 1 } };

    kalmanFilter.transitionMatrix = cv::Mat(4, 4, CV_32F, fltTransitionMatrixValues);

    /*
    kalmanFilter.statePre.at<float>(0) = 0;
    kalmanFilter.statePre.at<float>(1) = 0;
    kalmanFilter.statePre.at<float>(2) = 0;
    kalmanFilter.statePre.at<float>(3) = 0;
    */
                                                                           // research these !!!!!!!!!!!!!!!!!!!
    cv::setIdentity(kalmanFilter.measurementMatrix);                                //    
    cv::setIdentity(kalmanFilter.processNoiseCov, cv::Scalar::all(1));           // was 1e-4
    cv::setIdentity(kalmanFilter.measurementNoiseCov, cv::Scalar::all(1));         // was 10
    cv::setIdentity(kalmanFilter.errorCovPost, cv::Scalar::all(1));               // was 0.1
    
    cv::Mat image(600, 800, CV_8UC3, cv::Scalar::all(0));

    std::vector<cv::Point> predictedMousePositions;
    std::vector<cv::Point> actualMousePositions;
    std::vector<cv::Point> correctedMousePositions;

    cv::namedWindow("image");
    cv::setMouseCallback("image", mouseMoveCallback);

    while (true) {
        cv::Mat matPredicted = kalmanFilter.predict();

        cv::Point ptPredicted((int)matPredicted.at<float>(0), (int)matPredicted.at<float>(1));

        cv::Mat matActualMousePosition(2, 1, CV_32F, cv::Scalar::all(0));

        matActualMousePosition.at<float>(0, 0) = (float)ptActualMousePosition.x;
        matActualMousePosition.at<float>(1, 0) = (float)ptActualMousePosition.y;

        cv::Mat matCorrected = kalmanFilter.correct(matActualMousePosition);        // function correct() updates the predicted state from the measurement

        cv::Point ptCorrected((int)matCorrected.at<float>(0), (int)matCorrected.at<float>(1));

        predictedMousePositions.push_back(ptPredicted);
        actualMousePositions.push_back(ptActualMousePosition);
        correctedMousePositions.push_back(ptCorrected);

            // predicted, actual, and corrected are all now calculated, time to draw stuff

        drawCross(image, ptPredicted, SCALAR_BLUE);
        drawCross(image, ptActualMousePosition, SCALAR_WHITE);
        drawCross(image, ptCorrected, SCALAR_GREEN);
        
        for (int i = 0; i < predictedMousePositions.size() - 1; i++) {
            cv::line(image, predictedMousePositions[i], predictedMousePositions[i + 1], SCALAR_BLUE, 1);
        }

        for (int i = 0; i < actualMousePositions.size() - 1; i++) {
            cv::line(image, actualMousePositions[i], actualMousePositions[i + 1], SCALAR_WHITE, 1);
        }

        for (int i = 0; i < correctedMousePositions.size() - 1; i++) {
            cv::line(image, correctedMousePositions[i], correctedMousePositions[i + 1], SCALAR_GREEN, 1);
        }

        cv::imshow("image", image);         // show image
        
        cv::waitKey(10);                    // pause for a moment to get operating system to redraw the image

        image = cv::Scalar::all(0);         // blank the image for next time around
    }

    return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void mouseMoveCallback(int event, int x, int y, int flags, void* userData) {
    if (event == CV_EVENT_MOUSEMOVE) {
        std::cout << "mouse move at " << x << ", " << y << "\n";

        ptActualMousePosition.x = x;
        ptActualMousePosition.y = y;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawCross(cv::Mat &img, cv::Point center, cv::Scalar color) {
    cv::line(img, cv::Point(center.x - 5, center.y - 5), cv::Point(center.x + 5, center.y + 5), color, 2);
    cv::line(img, cv::Point(center.x + 5, center.y - 5), cv::Point(center.x - 5, center.y + 5), color, 2);

}




