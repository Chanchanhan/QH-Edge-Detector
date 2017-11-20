#ifndef _POINT_SET_H
#define _POINT_SET_H
#include <opencv2/core/core.hpp>
#include <vector>
namespace OD
{
	struct m_img_point_data{
	   cv::Point  m_img_point;
	   float data;
	   m_img_point_data(){}
	    m_img_point_data(cv::Point  p,float d){
	      m_img_point =p;
	      data= d;
	    }
	  };
	class PointSet
	{

	public:
	  void Clear(){
	    m_model_points.clear();
	    m_img_points.clear();
	    m_correspoinding_points.clear();
	    m_proj_points.clear();
	    m_img_points_f.clear();
	    
	  }
	public:
	  std::vector<cv::Point3f> m_model_points;
	  std::vector<cv::Point> m_img_points;
	  std::vector<cv::Point2f> m_img_points_f;
	  std::vector<cv::Point2f> m_correspoinding_points;
	  std::vector<cv::Point3f> m_proj_points;
	  cv::Mat m_fill_object;
	};
}

#endif