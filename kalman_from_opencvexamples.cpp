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
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 255.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);

cv::Point mousePos(0, 0);

///////////////////////////////////////////////////////////////////////////////////////////////////
int main(void) {
    
    cv::KalmanFilter kalmanFilter(4, 2, 0);

    float fltTransitionMatrixValues[4][4] = { { 1, 0, 1, 0 },
                                              { 0, 1, 0, 1 },
                                              { 0, 0, 1, 0 },
                                              { 0, 0, 0, 1 } };

    kalmanFilter.transitionMatrix = cv::Mat(4, 4, CV_32F, fltTransitionMatrixValues);
    
    cv::Mat matMeasurement(2, 1, CV_32F, cv::Scalar::all(0));

    kalmanFilter.statePre.at<float>(0) = (float)mousePos.x;
    kalmanFilter.statePre.at<float>(1) = (float)mousePos.y;
    kalmanFilter.statePre.at<float>(2) = 0;
    kalmanFilter.statePre.at<float>(3) = 0;


    setIdentity(kalmanFilter.measurementMatrix);
    setIdentity(kalmanFilter.processNoiseCov, cv::Scalar::all(1e-4));
    setIdentity(kalmanFilter.measurementNoiseCov, cv::Scalar::all(10));
    setIdentity(kalmanFilter.errorCovPost, cv::Scalar::all(.1));
    
    cv::Mat image(600, 800, CV_8UC3);

    std::vector<cv::Point> mousev, kalmanv;

    mousev.clear();
    kalmanv.clear();

    cv::namedWindow("image");
    cv::setMouseCallback("image", mouseMoveCallback);

    while (true) {
        cv::Mat matPrediction = kalmanFilter.predict();                      // function predict() computes a predicted state

        cv::Point predictedPoint((int)matPrediction.at<float>(0), (int)matPrediction.at<float>(1));     // this does not seem to be used ??

        matMeasurement.at<float>(0, 0) = (float)mousePos.x;
        matMeasurement.at<float>(1, 0) = (float)mousePos.y;

        cv::Mat estimated = kalmanFilter.correct(matMeasurement);        // function correct() updates the predicted state from the measurement

        cv::Point statePt((int)estimated.at<float>(0), (int)estimated.at<float>(1));

        cv::Point measPt((int)matMeasurement.at<float>(0, 0), (int)matMeasurement.at<float>(1, 0));

        cv::imshow("image", image);
        image = cv::Scalar::all(0);

        mousev.push_back(measPt);
        kalmanv.push_back(statePt);

        drawCross(image, statePt, SCALAR_RED);
        drawCross(image, measPt, SCALAR_WHITE);

        for (int i = 0; i < mousev.size() - 1; i++) {
            cv::line(image, mousev[i], mousev[i + 1], cv::Scalar(255, 255, 0), 1);
        }
        
        for (int i = 0; i < kalmanv.size() - 1; i++) {
            cv::line(image, kalmanv[i], kalmanv[i + 1], cv::Scalar(0, 155, 255), 1);
        }
        
        cv::waitKey(10);
    }

    return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void mouseMoveCallback(int event, int x, int y, int flags, void* userData) {
    if (event == CV_EVENT_MOUSEMOVE) {
        std::cout << "mouse move at " << x << ", " << y << "\n";

        mousePos.x = x;
        mousePos.y = y;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawCross(cv::Mat &img, cv::Point center, cv::Scalar color) {
    cv::line(img, cv::Point(center.x - 5, center.y - 5), cv::Point(center.x + 5, center.y + 5), color, 2);
    cv::line(img, cv::Point(center.x + 5, center.y - 5), cv::Point(center.x - 5, center.y + 5), color, 2);

}



