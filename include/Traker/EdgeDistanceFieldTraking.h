#ifndef _OPTIMIZER_H
#define _OPTIMIZER_H
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include "Traker/PointSet.hpp"
#include "Traker/Config.h"
#include "Traker/Model.h"
#include "Traker/Quaternion.h"
#include "Traker/Transformation.h"

#include"GLRenderer/include/glRenderer.h"

#include <fstream>

namespace OD
{


	

	class EdgeDistanceFieldTraking
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
	public:
		EdgeDistanceFieldTraking(const Config& config, const cv::Mat& initPose,const bool is_writer =false);
		~EdgeDistanceFieldTraking();
	public:
		cv::Mat optimizingLM(const cv::Mat& prePose,const cv::Mat& frame, const cv::Mat &locations, const int frameId);
		cv::Mat toComputePose(const Mat& prePose,const Mat& preFrame, const Mat& curFrame, const Mat& locations, const int frameId,const GLRenderer &render);
	private:
		static void lm(double *p, double* x, int m, int n, void* data);	
		static void jaclm(double *p, double *jac, int m, int n, void* data);
		void getDist(cv::Mat frame);
		void constructEnergyFunction(const cv::Mat frame, const cv::Mat prePose ,const cv::Mat &lastA,const int &lamda, cv::Mat &A, cv::Mat &b);
		void solveEnergyFunction();
		float computeEnergy(const cv::Mat& frame,const cv::Mat& pose);
		float nearestEdgeDistance(const cv::Point & point,const std::vector<m_img_point_data>  &edge_points,cv::Point &nPoint ,const bool printP=false);
		void UpdateStateLM(const cv::Mat &dx, const cv::Mat &pose_Old, cv::Mat &pose_New);
		void UpdateStateLM(const cv::Mat &dx, const cv::Mat &pose_Old, Transformation &transformation_New);
		int localChamferMatching(const Mat &distMap,const double *prevPose,const GLRenderer &renderer,const double K[9], double *pose,const int localSize);
		int localColorMatching(Mat frame,const Mat prevFrame,const  double *prevPose,const GLRenderer &renderer,const double K[9],
	double *pose, int localSize);
		void getMk();
		float getDistanceToEdege(const cv::Point& e1,const cv::Point& e2,const cv::Point& v);
		cv::Point getNearstPointLocation(const cv::Point &point);
	public:
		Data m_data;	
	private:
	  
		int nCtrPts, nReservedCtrPts;
		double(*ctrPts3D)[3];
		float *dist;
		vector<Point3f> contourPoints;
		int *_locations;
		int imgHeight,imgWidth;
		int _row;
		bool m_is_writer;
		std::ofstream m_outPose;
		Transformation m_Transformation;
		CameraCalibration m_calibration;

	};
}
#endif