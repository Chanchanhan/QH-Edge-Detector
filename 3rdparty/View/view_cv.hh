#pragma once 
#include <opencv2/opencv.hpp>
#include "view/view.hh"
#include "object/model.hh"

namespace tk {

class ViewCV: public View {
public:
	ViewCV(float fx, float fy, float cx, float cy, int width, int height);
	static ViewCV* Instance();

	void Draw(Model& model, Pose& pose, cv::Scalar color, cv::Mat& frame);
	void Project(Pose& pose, cv::Mat& model_points, cv::Mat& image_points);
	void Project(Pose& pose, std::vector<cv::Point3f>& pts, std::vector<cv::Point>& img_pts);
	void Project(Pose& pose, std::vector<cv::Point3f>& pts, std::vector<cv::Point2f>& img_pts);

	static ViewCV* instance;
};

} // namespace tk