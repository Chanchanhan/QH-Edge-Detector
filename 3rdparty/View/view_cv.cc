// copyright (c) shandong university. all rights reserved
// huanghone@sina.com

#include <glog/logging.h>
// #include "base/global_param.hh"
// #include "object/pose.hh"
#include "view/view_cv.hh"

namespace tk {

ViewCV* ViewCV::instance = NULL;

ViewCV* ViewCV::Instance() {
	tk::GlobalParam* gp = tk::GlobalParam::Instance();
	if (instance == NULL) instance = new ViewCV(gp->fx, gp->fy, gp->cx, gp->cy, gp->image_width, gp->image_height);
	return instance;
}

ViewCV::ViewCV(float fx, float fy, float cx, float cy, int width, int height) {
	width_ = width, height_ = height;

	k_ary[0] = fx, k_ary[1] = 0,  k_ary[2] = cx;
	k_ary[3] = 0,  k_ary[4] = fy, k_ary[5] = cy;
	k_ary[6] = 0,  k_ary[7] = 0,  k_ary[8] = 1;

	k_mat[0] = cv::Mat::zeros(3, 3, CV_32F);
	k_mat[0].at<float>(0, 0) = fx;
	k_mat[0].at<float>(0, 2) = cx;
	k_mat[0].at<float>(1, 1) = fy;
	k_mat[0].at<float>(1, 2) = cy;
	k_mat[0].at<float>(2, 2) = 1;

	k_ary[9] = fx/2.0f, k_ary[10] = 0,  k_ary[11] = cx/2.0f;
	k_ary[12] = 0,  k_ary[13] = fy/2.0f, k_ary[14] = cy/2.0f;
	k_ary[15] = 0,  k_ary[16] = 0,  k_ary[17] = 1;

	k_mat[1] = cv::Mat::zeros(3, 3, CV_32F);
	k_mat[1].at<float>(0, 0) = fx/2.0f;
	k_mat[1].at<float>(0, 2) = cx/2.0f;
	k_mat[1].at<float>(1, 1) = fy/2.0f;
	k_mat[1].at<float>(1, 2) = cy/2.0f;
	k_mat[1].at<float>(2, 2) = 1;

	k_ary[18] = fx/4.0f, k_ary[19] = 0,  k_ary[20] = cx/4.0f;
	k_ary[21] = 0,  k_ary[22] = fy/4.0f, k_ary[23] = cy/4.0f;
	k_ary[24] = 0,  k_ary[25] = 0,  k_ary[26] = 1;

	k_mat[2] = cv::Mat::zeros(3, 3, CV_32F);
	k_mat[2].at<float>(0, 0) = fx/4.0f;
	k_mat[2].at<float>(0, 2) = cx/4.0f;
	k_mat[2].at<float>(1, 1) = fy/4.0f;
	k_mat[2].at<float>(1, 2) = cy/4.0f;
	k_mat[2].at<float>(2, 2) = 1;

	scale = 0;
}

void ViewCV::Draw(Model& model, Pose& pose, cv::Scalar color, cv::Mat& frame) {
	cv::Mat visualable_model_points;
	model.GetVisualableVertices(pose, visualable_model_points);

	cv::Mat image_points;
	Project(pose, visualable_model_points, image_points);

	int size = image_points.cols/2;
	for (int i = 0; i < size; ++i) {
		cv::Point pt1(image_points.at<float>(0, 2*i), image_points.at<float>(1, 2*i));
		cv::Point pt2(image_points.at<float>(0, 2*i+1), image_points.at<float>(1, 2*i+1));
			
		if (PointInFrame(pt1) && PointInFrame(pt2)) {
			cv::line(frame, pt1, pt2, color);
		}
	}
}

void ViewCV::Project(Pose& pose, cv::Mat& model_points, cv::Mat& image_points) {
	CHECK(model_points.rows == 4) << "3d point must be homogeneous form";
	
	cv::Mat mv_mat(3, 4, CV_32FC1);
	pose.GetExtrinsicMat(mv_mat);

	image_points = k_mat[scale]*mv_mat*model_points;

	for (int i = 0; i < image_points.cols; ++i) {
		float dz = 1.0f/image_points.at<float>(2, i);
		image_points.at<float>(0, i) *= dz;
		image_points.at<float>(1, i) *= dz;
	}
}

void ViewCV::Project(Pose& pose, std::vector<cv::Point3f>& pts3d, std::vector<cv::Point>& pts2d) {
	cv::Mat mat3d(4, pts3d.size(), CV_32FC1);
	cv::Mat mat2d(3, pts3d.size(), CV_32FC1);

	for (int i = 0; i < (int)pts3d.size(); ++i) {
		mat3d.at<float>(0, i) = pts3d[i].x;
		mat3d.at<float>(1, i) = pts3d[i].y;
		mat3d.at<float>(2, i) = pts3d[i].z;
		mat3d.at<float>(3, i) = 1;
	}

	cv::Mat mv_mat(3, 4, CV_32FC1);
	pose.GetExtrinsicMat(mv_mat);

	mat2d = k_mat[scale]*mv_mat*mat3d;

	pts2d.clear();
	for (int i = 0; i < (int)pts3d.size(); ++i) {
		float dz = 1.0f/mat2d.at<float>(2,i);
		float x = mat2d.at<float>(0, i)*dz;
		float y = mat2d.at<float>(1, i)*dz;
		pts2d.push_back(cv::Point(x, y));
	}
}

void ViewCV::Project(Pose& pose, std::vector<cv::Point3f>& pts3d, std::vector<cv::Point2f>& pts2d) {
	cv::Mat mat3d(4, pts3d.size(), CV_32FC1);
	cv::Mat mat2d(3, pts3d.size(), CV_32FC1);

	for (int i = 0; i < (int)pts3d.size(); ++i) {
		mat3d.at<float>(0, i) = pts3d[i].x;
		mat3d.at<float>(1, i) = pts3d[i].y;
		mat3d.at<float>(2, i) = pts3d[i].z;
		mat3d.at<float>(3, i) = 1;
	}

	cv::Mat mv_mat(3, 4, CV_32FC1);
	pose.GetExtrinsicMat(mv_mat);

	mat2d = k_mat[scale]*mv_mat*mat3d;

	pts2d.clear();
	for (int i = 0; i < (int)pts3d.size(); ++i) {
		float dz = 1.0f/mat2d.at<float>(2,i);
		float x = mat2d.at<float>(0, i)*dz;
		float y = mat2d.at<float>(1, i)*dz;
		pts2d.push_back(cv::Point2f(x, y));
	}
}

} // namespace tk