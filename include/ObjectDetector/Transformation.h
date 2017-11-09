#ifndef TRANSFORMATION_H
#define TRANSFORMATION_H

#include<math.h>

#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>

#include "ObjectDetector/Quaternion.h"
#include "sophus/se3.h"
namespace OD{
class Transformation{
public:
  Transformation();
  Transformation(const cv::Mat &_pose);
  void rotateWithX(const float &angle);
  void rotateWithY(const float &angle);
  void rotateWithZ(const float &angle);
  void rotateWithR(const cv::Mat &Rotation);
  void translationWith(const float &_x,const float &_y,const float &_z);
  void setPoseFromTransformationMatrix(const cv::Mat &T);
  void setPoseFromTransformationMatrix(const Eigen::Matrix<double,4,4> &T);

  cv::Mat getDistortion(const cv::Mat &meanP);
  static cv::Mat getTransformationMatrix(const cv::Mat &pose);
  static cv::Mat getRotationMatrix(const cv::Vec3f &parametrization);
  static cv::Mat getRotationParametrization(const cv::Mat &rotMat);
  static cv::Mat getRotationMatrixb_so3(const float &x,const float &y, const float &z,float &theta,cv::Mat &so_x);
  static cv::Mat getTransformationMatrix(const cv::Vec3f &YDR,const cv::Vec3f &translation);
  cv::Mat transformationMatrix() ;
  void xTransformation(Transformation &_T);
  void xTransformation(const cv::Mat &Dx);
  void setPose(const cv::Mat &pose);
  float &x();
  float &y();
  float &z();
  float &u1();
  float &u2();
  float &u3();
public:  
  const cv::Mat &Pose();
  Quaternion &quaternion();
private:
  cv::Mat m_pose;
  Quaternion m_quaternion;
};
}



#endif
