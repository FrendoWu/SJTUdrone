#pragma once
#include "opencv2/aruco.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>


using namespace cv;

int main(int argc, char *argv[]) {
	VideoCapture inputVideo;
	int waitTime;
	inputVideo.open(0);
	inputVideo.set(CAP_PROP_FRAME_WIDTH, 1280);
	inputVideo.set(CAP_PROP_FRAME_HEIGHT, 720);
	waitTime = 10;

	double totalTime = 0;
	int totalIterations = 0;

	Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_ARUCO_ORIGINAL);
	Mat out;
	dictionary->drawMarker(100, 600, out, 5);

	Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
	detectorParams->minDistanceToBorder = 0;

	while (inputVideo.grab()) {
		Mat image, imageCopy;
		inputVideo.retrieve(image);
		double tick = (double)getTickCount();
		std::vector< int > ids;
		std::vector< std::vector< Point2f > > corners, rejected;
		std::vector< Vec3d > rvecs, tvecs;
		// detect markers and estimate pose
		aruco::detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);

		double currentTime = ((double)getTickCount() - tick) / getTickFrequency();
		totalTime += currentTime;
		totalIterations++;
		// draw results
		image.copyTo(imageCopy);
		if (ids.size() > 0)
		{
			aruco::drawDetectedMarkers(imageCopy, corners, ids);
			if (totalIterations % 30 == 0) {
				std::cout << "Detection Time = " << currentTime * 1000 << " ms "
					<< "(Mean = " << 1000 * totalTime / double(totalIterations) << " ms)" << std::endl;
			}
		}

		imshow("out", imageCopy);
		char key = (char)waitKey(waitTime);
		if (key == 27) break;
	}

	return 0;
}