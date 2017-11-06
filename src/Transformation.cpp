#include<glog/logging.h>

#include "ObjectDetector/Transformation.h"

using namespace OD;
Transformation::Transformation(){
  m_pose=cv::Mat::zeros(1,6,CV_32FC1);
}
Transformation::Transformation(const cv::Mat& _pose)
{
  m_pose=cv::Mat::zeros(1,6,CV_32FC1);
  m_pose=_pose.clone();
}
void Transformation::translationWith(const float& _x, const float& _y, const float& _z)
{
  x()+=_x;
  y()+=_y;
  z()+=_z;
}

const cv::Mat& Transformation::Pose()
{
  return m_pose;
}
void Transformation::setPose(const cv::Mat& pose)
{
  m_pose=pose.clone();
}

void Transformation::rotateWithR(const cv::Mat& Rotation)
{
  cv::Mat T = cv::Mat::eye(4,4,CV_32FC1);
  for(int c=0; c<3; c++)
  {
    for(int r=0; r<3; r++)
    {
      T.at<float>(r,c) = Rotation.at<float>(r,c);
    }    
  }
  T.at<float>(0,3)=0; T.at<float>(1,3)=0; T.at<float>(2,3) = 0;
  cv::Mat m_T= getTransformationMatrix(m_pose);
  m_T=T*m_T;
  setPoseFromTransformationMatrix(m_T);
  
}

void Transformation::xTransformation(Transformation& _T)
{

  cv::Mat m_T=transformationMatrix();
  m_T=_T.transformationMatrix()*m_T;
  setPoseFromTransformationMatrix(m_T);
}

void Transformation::xTransformation(const cv::Mat &Dx)
{
  cv::Mat m_T=transformationMatrix();
  float theta;
  cv::Mat so_x;
  cv::Mat rotMat=getRotationMatrixb_so3(Dx.at<float>(0,0),Dx.at<float>(0,1),Dx.at<float>(0,2),theta,so_x);
  cv::Mat V =cv::Mat(3,3,CV_32FC1);
  cv::Mat I3=cv::Mat::eye(3,3,CV_32FC1);
  cv::Mat t =cv::Mat(3,1,CV_32FC1);
  V=I3+so_x*(1-cos(theta))/(theta*theta)+(so_x*so_x)*(theta-sin(theta))/(theta*theta*theta);
  
  t.at<float>(0,0)=Dx.at<float>(0,3);
  t.at<float>(0,1)=Dx.at<float>(0,4);
  t.at<float>(0,2)=Dx.at<float>(0,5);
//   LOG(INFO)<<"t"<<t;
//   LOG(INFO)<<"V"<<V;
  t=V*t;
//   LOG(INFO)<<"t"<<t;
  cv::Mat T = cv::Mat::eye(4,4,CV_32FC1);
  for(int c=0; c<3; c++)
  {
    for(int r=0; r<3; r++)
    {
      T.at<float>(r,c) = rotMat.at<float>(r,c);
    }    
  }
  T.at<float>(0,3)=t.at<float>(0,0); T.at<float>(1,3)=t.at<float>(0,1); T.at<float>(2,3) = t.at<float>(0,2);
//   LOG(INFO)<<"T"<<T;
  setPoseFromTransformationMatrix(T*m_T);

}
cv::Mat Transformation::getRotationMatrixb_so3(const float &x,const float &y, const float &z,float &theta,cv::Mat &so_x)
{
 theta=sqrt(x*x+y*y+z*z);
 so_x=cv::Mat::zeros(3,3,CV_32FC1);
 cv::Mat I3=cv::Mat::eye(3,3,CV_32FC1);
 so_x.at<float>(1,2)=-x;
 so_x.at<float>(2,1)=x;
 so_x.at<float>(0,2)=y;
 so_x.at<float>(2,0)=-y;
 so_x.at<float>(0,1)=-z;
 so_x.at<float>(1,0)=z;
 cv::Mat rotMat=cv::Mat::zeros(3,3,CV_32FC1);
 rotMat=I3+so_x*sin(theta)/theta+(so_x*so_x)*(1-cos(theta))/(theta*theta);
 return rotMat;
}

Quaternion& Transformation::quaternion()
{
//   m_quaternion.SetEulerAngle(cv::Vec3f(u1(),u2(),u3()));
  return  m_quaternion;
}
float& Transformation::u1()
{
  return m_pose.at<float>(0,0);
}
float& Transformation::u2()
{
  return m_pose.at<float>(0,1);
}
float& Transformation::u3()
{
  return m_pose.at<float>(0,2);
}
float& Transformation::x()
{
  return m_pose.at<float>(0,3);
}
float& Transformation::y()
{
  return m_pose.at<float>(0,4);
}
float& Transformation::z()
{
  return m_pose.at<float>(0,5);
}

void Transformation::setPoseFromTransformationMatrix(const cv::Mat &T)
{
//  LOG(INFO)<<"in T"<<T;
 
 int startIndex=1;
 cv::Mat rotMat(3,3,CV_32FC1);
 cv::Mat roV(3,1,CV_32FC1);
 for(int c=0; c<3; c++)
 {
   for(int r=0; r<3; r++)
   {
     rotMat.at<float>(r,c) = T.at<float>(r,c);
   }    
 }
 roV=getRotationParametrization(rotMat);
 u1()=roV.at<float>(0,0); 
 u2()=roV.at<float>(1,0);
 u3()=roV.at<float>(2,0);
 x()=T.at<float>(0,3);
 y()=T.at<float>(1,3);
 z()=T.at<float>(2,3);
//  LOG(INFO)<<"out T"<<transformationMatrix();
}

void Transformation::rotateWithZ(const float &angle)
{
  cv::Mat T = getTransformationMatrix(cv::Vec3f (0,0,angle), cv::Vec3f(0,0,0));
  cv::Mat m_T= transformationMatrix();
  m_T=T*m_T;
  setPoseFromTransformationMatrix(m_T);
  
}
void Transformation::rotateWithY(const float &angle)
{
  cv::Mat T = getTransformationMatrix(cv::Vec3f (0,angle,0), cv::Vec3f(0,0,0));
  cv::Mat m_T= transformationMatrix();
  m_T=T*m_T;
  setPoseFromTransformationMatrix(m_T);
  
}
void Transformation::rotateWithX(const float &angle)
{
  cv::Mat T = getTransformationMatrix(cv::Vec3f (angle,0,0), cv::Vec3f(0,0,0));
  cv::Mat m_T= transformationMatrix();
  m_T=T*m_T;
  setPoseFromTransformationMatrix(m_T);  
}

cv::Mat Transformation::getRotationMatrix(const cv::Vec3f &parametrization)
{
    
  cv::Mat roV(3,1,CV_32FC1);
  roV.at<float>(0,0) = parametrization[0]; 
  roV.at<float>(1,0) = parametrization[1]; 
  roV.at<float>(2,0) = parametrization[2];
  cv::Mat rotMat(3,3,CV_32FC1);
  cv::Rodrigues(roV,rotMat);
  return rotMat;
}
cv::Mat Transformation::getRotationParametrization(const cv::Mat& rotMat)
{
  cv::Mat roV(3,1,CV_32FC1);
  cv::Rodrigues(rotMat,roV);
  return roV;
}

cv::Mat Transformation::transformationMatrix()
{

  return getTransformationMatrix(m_pose);

}


cv::Mat Transformation::getTransformationMatrix(const cv::Vec3f& parametrization, const cv::Vec3f& translation)
{
  cv::Mat roV(3,1,CV_32FC1);
  roV.at<float>(0,0) = parametrization[0]; 
  roV.at<float>(1,0) = parametrization[1]; 
  roV.at<float>(2,0) = parametrization[2];
  cv::Mat rotMat(3,3,CV_32FC1);
  cv::Mat T = cv::Mat::eye(4,4,CV_32FC1);
  cv::Rodrigues(roV,rotMat);
  for(int c=0; c<3; c++)
  {
    for(int r=0; r<3; r++)
    {
      T.at<float>(r,c) = rotMat.at<float>(r,c);
    }    
  }
  T.at<float>(0,3)=translation[0]; T.at<float>(1,3)=translation[1]; T.at<float>(2,3) = translation[2];
  return T;
}


cv::Mat Transformation::getTransformationMatrix(const cv::Mat &pose)
{

  cv::Mat rotMat(3,3,CV_32FC1);
  rotMat=getRotationMatrix(cv::Vec3f(pose.at<float>(0,0),pose.at<float>(0,1),pose.at<float>(0,2)));
  cv::Mat T = cv::Mat::eye(4,4,CV_32FC1);
  for(int c=0; c<3; c++)
  {
    for(int r=0; r<3; r++)
    {
      T.at<float>(r,c) = rotMat.at<float>(r,c);
    }    
  }
  T.at<float>(0,3)=pose.at<float>(0,3); T.at<float>(1,3)=pose.at<float>(0,4); T.at<float>(2,3) = pose.at<float>(0,5);

  return T;
}