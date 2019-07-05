#ifndef ImageToMat_h
#define ImageToMat_h
#include "common/common_utils/StrictMode.hpp"
STRICT_MODE_OFF
#ifndef RPCLIB_MSGPACK
#define RPCLIB_MSGPACK clmdep_msgpack
#endif // !RPCLIB_MSGPACK
#include "rpc/rpc_error.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "common/ImageCaptureBase.hpp"
STRICT_MODE_ON
#include "vehicles/multirotor/api/MultirotorApi.hpp"
#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"
#include <iostream>
#include <chrono>
#include <vector>

using namespace msr::airlib;

class ImageToMat
{
private:
	
	typedef ImageCaptureBase::ImageRequest ImageRequest;
	typedef ImageCaptureBase::ImageResponse ImageResponse;
	typedef ImageCaptureBase::ImageType ImageType;

	msr::airlib::MultirotorRpcLibClient client;

	cv::Mat img;

public:
	cv::Mat get_front_mat()
	{
		std::vector<ImageRequest> request = { ImageRequest(0, ImageType::Scene) };
		const std::vector<ImageResponse>& response = client.simGetImages(request);

		img = cv::Mat(response.at(0).height, response.at(0).width, CV_8SC3);
		img = cv::imdecode(response.at(0).image_data_uint8, 1);

		return img;
	}

	cv::Mat get_below_mat()
	{
		std::vector<ImageRequest> request = { ImageRequest(3, ImageType::Scene) };
		const std::vector<ImageResponse>& response = client.simGetImages(request);

		img = cv::Mat(response.at(0).height, response.at(0).width, CV_8SC3);
		img = cv::imdecode(response.at(0).image_data_uint8, 1);

		return img;
	
	}
	//////
	std::vector<ImageResponse> get_depth_mat()
	{
		std::vector<ImageRequest> request = { ImageRequest(0, ImageType::DepthPerspective,true) };
		const std::vector<ImageResponse>& response = client.simGetImages(request);
		return response;
	}
};

#endif
