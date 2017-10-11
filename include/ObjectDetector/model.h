#ifndef _MODEL_H
#define _MODEL_H

#include "tools/glm.h"
#include <string>
#include <opencv2/core.hpp>
#include "edge/CameraCalibration.h"
#include "edge/Model_Config.h"
#include "ObjectDetector/PointSet.hpp"
namespace OD
{

	class Model
	{
	public:
		Model(const Config& config);
		~Model();
	public:
		void LoadGLMModel(const std::string& filename);
		PointSet& GetVisibleModelPointsCV(const cv::Mat& prepose, int pointnum);
		
		void GetImagePoints(const cv::Mat& prepose, PointSet& pointset);
		
		void DisplayCV(const cv::Mat& pose, cv::Mat& frame);
		void DisplayGL(const cv::Mat& pose);
	public:
		void computeExtrinsicByEuler(cv::Mat* mvMatrix, float& _x, float& _y, float& _z, float& _rx, float &_ry, float &_rz);
		void FilterModel(const cv::Mat& prepose,int pointnum);
		
		void InitPose(const cv::Mat& initPose);
		void getIntrinsic(cv::Mat& intrinsic) const;
		
		cv::Mat GetPoseMatrix();
		GLMmodel* GetObjModel();
		cv::Mat m_rvec;
		cv::Mat m_tvec;
	private:
		GLMmodel* m_model;
		ED::CameraCalibration m_calibration;
		int m_width;
		int m_height;
		PointSet m_point_set;
		float m_radius;
		cv::Mat m_bb_points;
		
		
	};
}
#endif