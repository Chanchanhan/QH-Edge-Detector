#pragma once

#include <opencv2/opencv.hpp>
#include "view/view.hh"

namespace tk {

class Pose;
class Model;

class ViewGL: public View {
public:
	ViewGL() {}

	void Draw(Model& model, Pose& pose, cv::Scalar color, cv::Mat& frame);
	
};

} // namespace tk