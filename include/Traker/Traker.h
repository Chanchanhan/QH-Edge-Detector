#ifndef _TRAKER_H
#define _TRAKER_H
#include <opencv2/core.hpp>
#include "Traker/PointSet.hpp"
#include "Traker/Model.h"
#include "Traker/Quaternion.h"
#include "Traker/Transformation.h"
#include "GLRenderer/include/glRenderer.h"
#include <fstream>

namespace OD
{
	struct Data
	{
	  PointSet m_pointset;
		Model* m_model;
	};

	

	class Traker
	{
	public:
		Traker( const float *initPose,const bool is_writer);
		~Traker();
	public:
		int toTrack(const float * prePose,const cv::Mat& curFrame, const int & frameId,const GLRenderer &glrender, float * _newPose ,float &finalE2);
		int toTrack2(const float * prePose,const cv::Mat& curFrame, const int & frameId,const GLRenderer &glrender, float * _newPose ,float &finalE2);

	private:
		int edfTracker(const float * prePose,const  int NLrefine, float* newPose);
		void constructEnergyFunction(const cv::Mat frame, const float* prePose ,const cv::Mat &lastA,const int &lamda, cv::Mat &A, cv::Mat &b);
		void constructEnergyFunction(const cv::Mat distFrame,const float* prePose,const cv::Mat &lastA,const int &lamda,const std::vector<cv::Point3d>  &countour_Xs,const std::vector<cv::Point2d>  &countour_xs, cv::Mat &A, cv::Mat &b);
		void constructEnergyFunction2(const cv::Mat frame, const float* prePose ,const cv::Mat &lastA,const int &lamda, cv::Mat &A, cv::Mat &b);
		void constructEnergyFunction2(const cv::Mat distFrame,const float* prePose,const cv::Mat &lastA,const int &lamda,const std::vector<cv::Point3d>  &countour_Xs,const std::vector<cv::Point2d>  &countour_xs, cv::Mat &A, cv::Mat &b);
		void solveEnergyFunction();
		float computeEnergy(const cv::Mat& distFrame,const float * pose);
		float computeEnergy(const cv::Mat& distFrame,const std::vector<cv::Point3d>  &countour_Xs,const std::vector<cv::Point2d>  &countour_xs);
		float computeEnergy(const cv::Mat& distFrame,const float * pose, std::vector<cv::Point3d>  &countour_Xs, std::vector<cv::Point2d>  &countour_xs);
		float nearestEdgeDistance(const cv::Point & point,const std::vector<m_img_point_data>  &edge_points,cv::Point &nPoint ,const bool printP=false);
		void UpdateStateLM(const cv::Mat &dx, const float * pose_Old, float * pose_New);
		void UpdateStateLM(const cv::Mat &dx, const float * pose_Old, Transformation &transformation_New);
		void getCoarsePoseByPNP(const float *prePose,const cv::Mat &distMap,float *coarsePose);
		void getCoarsePoseByPNP(const float *prePose, float *coarsePose,std::vector<cv::Point3d> &objectPoints, std::vector<cv::Point2d> &imagePoints);
		void getMk();
		void getDistMap(const cv::Mat &frame);
		void updateState(const cv::Mat&distFrame, const Mat& dX, const Transformation& old_Transformation, Transformation& new_transformation,float &e2_new);
		void updateState2(const cv::Mat&distFrame, const Mat& dX, const Transformation& old_Transformation, Transformation& new_transformation,float &e2_new);
		float getDistanceToEdege(const cv::Point& e1,const cv::Point& e2,const cv::Point& v);
		void getNearImgPoint(const std::vector<cv::Point2d> &imagePoints,std::vector<cv::Point2d> &nearPoints);
		cv::Point getNearstPointLocation(const cv::Point &point);
		
	public:
		Data m_data;	
	private:
		int N_Points;
		int m_frameId;
		int *_locations;
		float* dist;
		int imgHeight,imgWidth;
		cv::Mat mFrame;
		cv::Mat distFrame,locationsMat;
		double final_e;
		bool m_is_writer;
		std::ofstream m_outPose;
		Transformation m_Transformation;
		CameraCalibration m_calibration;

	};
}
#endif