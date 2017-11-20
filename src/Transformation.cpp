#include<glog/logging.h>
#include<math.h>
#include "ObjectDetector/Transformation.h"

using namespace OD;
Transformation::Transformation(){
  m_pose=new float[6];
}
Transformation::Transformation(const cv::Mat& _pose)
{
  m_pose=new float[6];
  for(int i =0;i<6;i++){
    m_pose[i]=_pose.at<float>(i,0);
  }
}
void Transformation::translationWith(const float& _x, const float& _y, const float& _z)
{
  x()+=_x;
  y()+=_y;
  z()+=_z;
}
cv::Mat Transformation::M_Pose()
{
  cv::Mat pose(1,6,CV_32FC1);
  for(int i=0;i<6;i++){
    pose.at<float>(0,i)=m_pose[i];
  }
  return pose;
}
cv::Mat Transformation::getMatPose(const float* pose)
{

  cv::Mat Mat_Pose(1,6,CV_32FC1);
  for(int i=0;i<6;i++){
    Mat_Pose.at<float>(0,i)=pose[i];
  }
  return Mat_Pose;
}

const float* Transformation::Pose() const
{
  return m_pose;
}

void Transformation::setPose(const cv::Mat& pose)
{
//   m_pose=pose.clone();
  std::cout<<"setPose";
  for(int i =0;i<6;i++){
    m_pose[i]=pose.at<float>(i,0);
    std::cout<<"  "<<m_pose[i];
  }
}
void Transformation::setPose(const float * pose,const bool check )
{
  if(check){
    if(!toMove(pose))	{
      LOG(WARNING)<<"Not allowed to move ,new T is too big!";
      return;
      
    }
  }
  memcpy(m_pose,pose,sizeof(float)*6);
//   for(int i=0;i<6;i++){
//    m_pose[i]=pose[i];
//   }
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
//   return m_pose.at<float>(0,0);
   return m_pose[0];
}
float& Transformation::u2()
{
  return m_pose[1];
//   return m_pose.at<float>(0,1);
}
float& Transformation::u3()
{
   return m_pose[2];
//   return m_pose.at<float>(0,2);
}
float& Transformation::x()
{
   return m_pose[3];
//   return m_pose.at<float>(0,3);
}
float& Transformation::y()
{
   return m_pose[4];
//   return m_pose.at<float>(0,4);
}
float& Transformation::z()
{
   return m_pose[5];
//   return m_pose.at<float>(0,5);
}
void Transformation::setPoseFromTransformationMatrix(const Eigen::Matrix< double, int(4), int(4) >& T)
{
 cv::Mat rotMat(3,3,CV_32FC1);
 cv::Mat roV(3,1,CV_32FC1);
 for(int c=0; c<3; c++)
 {
   for(int r=0; r<3; r++)
   {
     rotMat.at<float>(r,c) = T(r,c);
   }    
 }
 roV=getRotationParametrization(rotMat);
 u1()=roV.at<float>(0,0); 
 u2()=roV.at<float>(1,0);
 u3()=roV.at<float>(2,0);
 x()=T(0,3);
 y()=T(1,3);
 z()=T(2,3);
}

void Transformation::setPoseFromTransformationMatrix(const cv::Mat &T)
{
//  LOG(INFO)<<"in T"<<T;
 
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


cv::Mat Transformation::getTransformationMatrix(const float *pose)
{

  cv::Mat rotMat(3,3,CV_32FC1);
  rotMat=getRotationMatrix(cv::Vec3f(pose[0],pose[1],pose[2]));
  cv::Mat T = cv::Mat::eye(4,4,CV_32FC1);
  for(int c=0; c<3; c++)
  {
    for(int r=0; r<3; r++)
    {
      T.at<float>(r,c) = rotMat.at<float>(r,c);
    }    
  }
  T.at<float>(0,3)=pose[3]; T.at<float>(1,3)=pose[4]; T.at<float>(2,3) = pose[5];

  return T;
}

bool Transformation::toMove(const float* pose, const float maxTheta, const float maxTran)
{
  float theta,tran;
  getSub2Pose(pose,theta,tran);
  LOG(WARNING)<<"theta = "<<theta<<" tran = "<<tran;
  if(std::isnan(theta)||std::isnan(tran)){
    return false;
  }
  return (theta<maxTheta&&tran<maxTran);
}

void  Transformation::getSub2Pose(const float* pose, float &theta, float &tran)
{
  	
  Eigen::Vector3d v3d1(m_pose[0],m_pose[1],m_pose[2]),v3d2(pose[0],pose[1],pose[2]),v3d;     

  Sophus::SO3 S01=Sophus::SO3::exp(v3d1),S02=Sophus::SO3::exp(v3d2);
  S01=S02.inverse()*S01;
  v3d=S01.log();
  theta = sqrt(v3d[0]*v3d[0]+v3d[1]*v3d[1]+v3d[2]*v3d[2]);
  tran =sqrt(pow(m_pose[3]-pose[3],2)+pow(m_pose[4]-pose[4],2)+pow(m_pose[5]-pose[5],2));
}
