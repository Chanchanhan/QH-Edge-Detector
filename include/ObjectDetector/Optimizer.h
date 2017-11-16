#ifndef _OPTIMIZER_H
#define _OPTIMIZER_H
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include "ObjectDetector/PointSet.hpp"
#include "ObjectDetector/Model_Config.h"
#include "ObjectDetector/Model.h"
#include "ObjectDetector/Quaternion.h"
#include "ObjectDetector/Transformation.h"

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
		Optimizer(const Config& config, const float */* const cv::Mat& */initPose, bool is_writer);
		~Optimizer();
	public:
		void optimizingLM(const float * prePose,const cv::Mat& curFrame,const cv::Mat& distFrame, const cv::Mat &locations, const int frameId,float * _newPose );
	      
	private:
		int edfTracker(const float * prePose,const cv::Mat& distFrame,const  int NLrefine, float* newPose);
		void constructEnergyFunction(const cv::Mat frame, const float* prePose ,const cv::Mat &lastA,const int &lamda, cv::Mat &A, cv::Mat &b);
		void solveEnergyFunction();
		float computeEnergy(const cv::Mat& frame,const float * pose);
		float nearestEdgeDistance(const cv::Point & point,const std::vector<m_img_point_data>  &edge_points,cv::Point &nPoint ,const bool printP=false);
		void UpdateStateLM(const cv::Mat &dx, const float * pose_Old, float * pose_New);
		void UpdateStateLM(const cv::Mat &dx, const float * pose_Old, Transformation &transformation_New);
		void getCoarsePoseByPNP(const float *prePose,const cv::Mat &distMap,float *coarsePose);
		void getMk();
		float getDistanceToEdege(const cv::Point& e1,const cv::Point& e2,const cv::Point& v);
		cv::Point getNearstPointLocation(const cv::Point &point);
		
	public:
		Data m_data;	
	private:
		int *_locations;
		float* dist;
		int imgHeight,imgWidth;
		cv::Mat mFrame;
		double final_e;
		bool m_is_writer;
		std::ofstream m_outPose;
		Transformation m_Transformation;
		CameraCalibration m_calibration;

	};
}
#endif