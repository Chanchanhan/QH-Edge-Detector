#ifndef _OPTIMIZER_H
#define _OPTIMIZER_H
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include "ObjectDetector/PointSet.hpp"
#include "ObjectDetector/Model_Config.h"
#include "ObjectDetector/Model.h"
#include "ObjectDetector/Quaternion.h"
#include <fstream>

namespace OD
{
	struct Data
	{
		PointSet m_pointset;
		Model* m_model;
// 		Correspondence* m_correspondence;
		int m_n;

		cv::Mat m_frame;

		//for robust m-estimation
		float m_weight[1000];
		float m_residual[1000];
		float m_jac[1000][6];
	};

	

	class Optimizer
	{
	public:
		Optimizer(const Config& config, const cv::Mat& initPose, bool is_writer);
		~Optimizer();
	public:
		cv::Mat optimizingLM(const cv::Mat& prePose,const cv::Mat& frame, const int frameId);
	      
	private:
		static void lm(double *p, double* x, int m, int n, void* data);	
		static void jaclm(double *p, double *jac, int m, int n, void* data);
		void constructEnergyFunction(const cv::Mat prePose,const std::vector<cv::Point> &nPoints, const std::vector<m_img_point_data>  &m_img_points_data ,const cv::Mat &lastA,const int &lamda, cv::Mat &A, cv::Mat &b);
		void solveEnergyFunction();
		float computeEnergy(const cv::Mat& frame,const cv::Mat& pose,const std::vector<m_img_point_data> m_img_points_data,std::vector<cv::Point> &nPoints,const bool printP =false);
		float nearestEdgeDistance(const cv::Point & point,const std::vector<m_img_point_data>  &edge_points,cv::Point &nPoint ,const bool printP=false);
		void UpdateStateLM(const cv::Mat &dx, const cv::Mat &pose_Old, cv::Mat &pose_New);
		void getMk();
	public:
		Data m_data;	
	private:

		bool m_is_writer;
		std::ofstream m_outPose;
		CameraCalibration m_calibration;

	};
}
#endif