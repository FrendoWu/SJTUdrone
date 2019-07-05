#pragma once
using namespace ell;
#define V2SP Point2f p3,Point2f p2,Point2f p1,Point2f p4,Point2f p5,Point2f p6
float value4SixPoints(V2SP);
Point2f lineCrossPoint(Point2f l1p1, Point2f l1p2, Point2f l2p1, Point2f l2p2 );
void point2Mat(Point2f p1, Point2f p2, float mat[2][2]);

//for Validate
void DrawDetectedEllipses(Mat3b& output, std::vector<ell::Ellipse>& ellipses, 
						  int iTopN=0, int thickness=2);
void SaveEllipses(const std::string& fileName, const std::vector<ell::Ellipse>& ellipses);
void LoadGT(std::vector<ell::Ellipse>& gt, const std::string& sGtFileName,
			bool bIsAngleInRadians = true);
bool LoadTest(std::vector<ell::Ellipse>& ellipses, const std::string& sTestFileName,
			  std::vector<double>& times, bool bIsAngleInRadians = true);
bool TestOverlap(const Mat1b& gt, const Mat1b& test, float th);
int Count(const std::vector<bool> v);
std::vector<float> Evaluate(const std::vector<ell::Ellipse>& ellGT,
					   const std::vector<ell::Ellipse>& ellTest,
					   const float th_score, const Mat3b& img);
// show Result
void showResult(std::vector<double> timesAndRecognize);
std::vector<float> getRecognizeResult(std::string sWorkingDir, std::string imagename,
	std::vector<ell::Ellipse> ellsCned,
								 float fThScoreScore, bool showpic=false);
