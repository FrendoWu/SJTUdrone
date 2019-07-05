#include "EDLibCommon.h"
#include "tools.h"
#include "CNEllipseDetector.h"
#include <direct.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//Parameters Settings
float	fThScoreScore = 0.55f;	//
float	fMinReliability	= 0.4f;	// Const parameters to discard bad ellipses 0.4
float	fTaoCenters = 0.05f;
int		ThLength = 16;
float	MinOrientedRectSide = 3.0f;
int 	iNs = 16;//
float	scale = 1.0f;

vector<Ellipse> OnImage(cv::Mat& _image) {
	/*string sWorkingDir = "D:\\git\\Õ÷‘≤ºÏ≤‚\\";
	string imagename = "img_242.png";

	string filename = sWorkingDir +  imagename;*/

	// Read image
	Mat3b image = _image;
	Size sz = image.size();

	// Convert to grayscale
	Mat1b gray;
	cvtColor(image, gray, CV_BGR2GRAY);

	// Parameters Settings (Sect. )
	int		iThLength = ThLength;
	float	fThObb = MinOrientedRectSide;
	float	fThPos = 1.0f;
	float	fMaxCenterDistance = sqrt(float(sz.width*sz.width + sz.height*sz.height)) * fTaoCenters;

	// Other constant parameters settings.
	// Gaussian filter parameters, in pre-processing
	Size	szPreProcessingGaussKernelSize	= Size(5,5);
	double	dPreProcessingGaussSigma		= 1.0;

	float	fDistanceToEllipseContour		= 0.1f;	// (Sect. - Validation)

	CNEllipseDetector cned;
	// Initialize Detector with selected parameters
	cned.SetParameters	(szPreProcessingGaussKernelSize,dPreProcessingGaussSigma,		
		fThPos,fMaxCenterDistance,iThLength,fThObb,fDistanceToEllipseContour,		
		fThScoreScore,fMinReliability,		
		iNs );

	// Detect
	vector<Ellipse> ellsYaed;
	cned.Detect(gray.clone(), ellsYaed);

	vector<double> times = cned.GetTimes();
	cout << "--------------------------------" << endl;
	cout << "Execution Time: " << endl;
	cout << "Edge Detection: \t" << times[0] << endl;
	cout << "Pre processing: \t" << times[1] << endl;
	cout << "Grouping:       \t" << times[2] << endl;
	cout << "Estimation:     \t" << times[3] << endl;
	cout << "Validation:     \t" << times[4] << endl;
	cout << "Clustering:     \t" << times[5] << endl;
	cout << "--------------------------------" << endl;
	cout << "Total:	         \t" << cned.GetExecTime() << endl;
	cout << "--------------------------------" << endl;

	vector<Ellipse> gt;
	//LoadGT(gt, sWorkingDir + "/gt/" + "gt_" + imagename + ".txt", false); // Prasad is in radians(ª°∂»)

	Mat3b resultImage = image.clone();

	// Draw GT ellipses
	for (unsigned i = 0; i < gt.size(); ++i) {
		Ellipse& e = gt[i];
		Scalar color(0, 0, 255);
		ellipse(resultImage, Point(cvRound(e._xc), cvRound(e._yc)), Size(cvRound(e._a), cvRound(e._b)), e._rad*180.0 / CV_PI, 0.0, 360.0, color, 3);
	}

	DrawDetectedEllipses(resultImage, ellsYaed);

	Mat3b res = image.clone();
	vector<float> result;
	result = Evaluate(gt, ellsYaed, fThScoreScore, res);
	cout << "F-measure : " << result[5] << endl;
	imshow("CNED", resultImage);
	waitKey();
	return ellsYaed;
}


//int main(int argc, char** argv) {
//	OnImage();
//	return 0;	   
//}