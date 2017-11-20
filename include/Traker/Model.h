#ifndef _MODEL_H
#define _MODEL_H

#include <string>
#include <opencv2/core.hpp>
#include "Traker/CameraCalibration.h"
#include "Traker/Config.h"
#include "Traker/PointSet.hpp"
#include "GLRenderer/include/glm.h"

namespace OD
{
	struct Line{
	 cv::Mat v1,v2;
	};
	class Model
	{

// 	struct Triangle{
// 	 int v1,v2;
// 	 std::vector<Vec3f> points;
// 	 bool visible;
// 	};
	public:
		Model(const Config& config);
		~Model();
	public:
		void LoadGLMModel(const std::string& filename);
		PointSet& GetVisibleModelPointsCV(const cv::Mat& prepose, int pointnum);
		void GetImagePoints(const float * prepose, PointSet& pointset);

		void setPointSet();
		void DisplayCV(const float * pose, cv::Mat& frame);
		void DisplayGL(const cv::Mat& pose);
		void DisplayLine(const cv::Point& p1,const cv::Point& p2, cv::Mat& frame,const float &radius);
		void computeExtrinsicByEuler(cv::Mat* mvMatrix, float& _x, float& _y, float& _z, float& _rx, float &_ry, float &_rz);
		void FilterModel(const cv::Mat& prepose,int pointnum);
		void getVisitLines(const cv::Mat pose);
		void getIntrinsic(cv::Mat& intrinsic) const;
		void setVisibleLinesAtPose(const float * pose);
		void getVisualableVertices(const float * pose, cv::Mat& vis_vertices);
		const cv::Mat& getIntrinsic()  const;
		const std::vector<Line> & getMyLines();
		
		
		const cv::Mat& getPos() const;
		Point X_to_x(Point3f X,Mat extrisic);

		cv::Mat GetPoseMatrix();
		cv::Mat GetPoseMatrix(cv::Mat pose);
		cv::Mat GetPoseMatrix(const float * pose);
	public:

		GLMmodel* GetObjModel();
	private:
		cv::Mat modelPos;
		std::vector<Line> myLines;
		GLMmodel* m_model;
		OD::CameraCalibration m_calibration;
		int m_width;
		int m_height;
		PointSet m_point_set;
		float m_radius;
		cv::Mat m_bb_points;
		cv::Mat intrinsic;
		
	private:
		cv::Point3f getPos_E(int e);
		bool checkPointInTrinangle(const cv::Point p,const cv::Point p1, const cv::Point p2,const cv::Point p3);
		void getVisibleLines();
		int crossProductNorm(const cv::Point &p,const cv::Point &p1);
		bool isLineVisible(const cv::Point &v1,const cv::Point &v2,const PointSet &point_set);
		bool isPointVisible(const cv::Point &v1,const PointSet &point_set);
		bool isSameNormal(const float *n1,const float *n2);
		

	};
}
#endif