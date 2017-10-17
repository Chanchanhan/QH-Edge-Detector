#include <fstream>      //C++
#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include "ObjectDetector/Optimizer.h"
using namespace OD;
const int MAX_ITERATIN_NUM =100;
const float THREHOLD_ENERGY = 9000.0f;
const float STEP_LENGTH = 0.00001f;
static void printMat(std::string name, cv::Mat M){
  cout<<name<<std::endl;;
  for(int i=0;i<M.rows;i++){
    printf("i %d :",i );
    for(int j=0;j<M.cols;j++){
      printf("%f ",M.at<float>(i,j));
    }
    printf("\n");
  }
}
Optimizer::Optimizer(const Config& config, const Mat& initPose, bool is_writer)
{
  m_data.m_model = new Model(config);
  //m_data.m_model->LoadGLMModel(config.filename.c_str());
//   m_data.m_correspondence = new Correspondence(config.width,config.height);
//   m_data.m_correspondence->m_lineBundle.create(R,2*L+1,CV_8UC3);
//   m_data.m_correspondence->m_hsv.create(R,2*L+1,CV_8UC3);
  m_calibration = config.camCalibration;
  m_data.m_model->InitPose(initPose);
  m_is_writer = is_writer;
  if(is_writer)
  {
    //record the pose
    std::string poseFile = "output/out_pose.txt";
    m_outPose.open(poseFile);
    if( !m_outPose.is_open() )
    {
      printf("Cannot write the pose\n");
      return;      
    }
  }
}
Optimizer::~Optimizer()
{
  if(m_is_writer)
  {
    m_outPose.close();    
  }
}

cv::Mat Optimizer::optimizingLM(const Mat& prePose, const Mat& frame, const int frameId)
{
  int64 time0 = cv::getTickCount();
  {
    std::vector<m_img_point_data> m_img_points_data;  
    for(int i=0;i<frame.rows;i++){
      for(int j=0;j<frame.cols;j++){
	if(frame.at<float>(i,j)>1e0){
	  m_img_points_data.push_back(m_img_point_data(cv::Point(i,j),frame.at<float>(i,j)));
	}      
      }
    }
    std::vector<cv::Point> nPoints;

    cv::Mat _prePose =prePose.clone();
    int itration_num=0;
    cv::Mat newPose=cv::Mat::zeros(1,6,CV_32FC1);

    float e2 = computeEnergy(_prePose,m_img_points_data,nPoints);

    while(++itration_num<MAX_ITERATIN_NUM){      
      cv::Mat A= cv::Mat::zeros(6,6,CV_32FC1),b= cv::Mat::zeros(6,1,CV_32FC1);
      if(e2<THREHOLD_ENERGY){
	break;
      }
      constructEnergyFunction(nPoints,A,b);
      Mat A_inverse ;
      cv::invert(A,A_inverse);
    //  printMat("b",b);
    //  printMat("A_inverse",A_inverse);
      Mat dX = A_inverse*b*STEP_LENGTH;
//       printMat("dX",dX);
      UpdateStateLM(dX,_prePose,newPose);
      std::vector<cv::Point> new_nPoints;
      float e2_new = computeEnergy(newPose,m_img_points_data,new_nPoints);
      printf("e2 :%f e2_new:%f \n",e2,e2_new);
      if(e2_new<e2){
	_prePose=newPose.clone();
	e2= e2_new;
	printf("a little optimize\n");
      }
      if(e2_new<THREHOLD_ENERGY){
	printf("succees optimize!\n");
	return newPose;
      }
      
    }
    printf("to much itration_num!\n");
    return _prePose;
    
  }
  
  int64 teim0 = cv::getTickCount();
  
  //printf("solving time:%lf\n",(time_s1-time_s)/cv::getTickFrequency());
}

void Optimizer::constructEnergyFunction(const std::vector<cv::Point> &nPoints , cv::Mat &A, cv::Mat &b){
  cv::Mat j_X_Pose= cv::Mat::zeros(2,6,CV_32FC1);
  cv::Mat j_Energy_X=cv::Mat::zeros(1,2,CV_32FC1);
  float coffes = 1.0f/nPoints.size();
  for(int i=1;i<=m_data.m_model->GetObjModel()->numvertices;i++){
    cv::Mat _j_X_Pose = cv::Mat::zeros(2,6,CV_32FC1);
    cv::Mat _j_Energy_X=cv::Mat::zeros(1,2,CV_32FC1);
    float _x=m_data.m_model->GetObjModel()->vertices[i*3+0];
    float _y=m_data.m_model->GetObjModel()->vertices[i*3+1];
    float _z=m_data.m_model->GetObjModel()->vertices[i*3+2];
    _j_X_Pose.at<float>(0,0)+=m_calibration.fx()/_z; 
    _j_X_Pose.at<float>(0,2)+=-m_calibration.fx()*_x/(_z*_z);
    _j_X_Pose.at<float>(0,3)+=-m_calibration.fx()*_x*_y/(_z*_z);
    _j_X_Pose.at<float>(0,4)+=m_calibration.fx()*(1+_x*_x/(_z*_z));
    _j_X_Pose.at<float>(0,5)+=m_calibration.fx()*_y/_z;

    
    _j_X_Pose.at<float>(1,1)+=m_calibration.fy()/_z;      
    _j_X_Pose.at<float>(1,2)+=-m_calibration.fy()*_y/(_z*_z);    
    _j_X_Pose.at<float>(1,3)+=-m_calibration.fy()*(1+_y*_y*(1/(_z*_z)));
    _j_X_Pose.at<float>(1,4)+=-m_calibration.fy()*_x*_y/(_z*_z);
    _j_X_Pose.at<float>(1,5)+=m_calibration.fy()*_x/_z;
    
    _j_Energy_X.at<float>(0,0)=2*(m_data.m_pointset.m_img_points[i-1].x - nPoints[i-1].x);
    _j_Energy_X.at<float>(0,1)=2*(m_data.m_pointset.m_img_points[i-1].y - nPoints[i-1].y);
    
    _j_X_Pose*=coffes;
    _j_Energy_X*=coffes;
    _j_X_Pose/=100000;
    _j_Energy_X/=100;
    Mat _J=_j_Energy_X*_j_X_Pose;
    Mat _J_T;
    cv::transpose(_J,_J_T);
    b+=_J_T*( (m_data.m_pointset.m_img_points[i-1].x - nPoints[i-1].x)*(m_data.m_pointset.m_img_points[i-1].x - nPoints[i-1].x)
	   +(m_data.m_pointset.m_img_points[i-1].y - nPoints[i-1].y)*(m_data.m_pointset.m_img_points[i-1].y - nPoints[i-1].y));
//     printf("%d : point  %d %d , npoint %d %d\n",i,m_data.m_pointset.m_img_points[i-1].x,m_data.m_pointset.m_img_points[i-1].y,nPoints[i-1].x,nPoints[i-1].y);

    j_X_Pose+=_j_X_Pose;
    j_Energy_X+=_j_Energy_X;
    
  }
  Mat J(6,6,CV_32FC1),J_T(6,6,CV_32FC1);
  Mat I = Mat::eye(6,6,CV_32FC1);
  J=j_Energy_X*j_X_Pose;  
  cv::transpose(J,J_T);
  float u=0.5;
  A= J_T*J+I*u;//6*6
  
  
}

float Optimizer::computeEnergy(const cv::Mat& pose, const std::vector<m_img_point_data> m_img_points_data,std::vector<cv::Point> &nPoints )
{
  m_data.m_model->GetImagePoints(pose, m_data.m_pointset);
  float energy =0;
  float _dsize =1.0f/ m_data.m_pointset.m_img_points.size();
  for(auto point:m_data.m_pointset.m_img_points){
    cv::Point nPoint;
    energy+=nearestEdgeDistance(point,m_img_points_data,nPoint);
//       printf("point  %d %d , npoint %d %d\n",point.x,point.y,nPoint.x,nPoint.y);
    nPoints.push_back(nPoint);
  }
  energy*=_dsize;
//   printf("energy = %f \n",energy);
  return energy;
  
}

float Optimizer::nearestEdgeDistance(const cv::Point & point,const std::vector<m_img_point_data>  &edge_points,cv::Point &nPoint)
{
  assert(edge_points.size()>0);  
  float nearstD = -1;
  int ni=0,nj=0;
  
  for(auto edge_point:edge_points){
    float temp_D= (edge_point.m_img_point.x-point.x)*(edge_point.m_img_point.x-point.x)+(edge_point.m_img_point.y-point.y)*(edge_point.m_img_point.y-point.y);
    if(nearstD>temp_D||nearstD==-1){
      nearstD = temp_D;
      nPoint.x =edge_point.m_img_point.x,nPoint.y=edge_point.m_img_point.y;      
    }
  }
//   J.at<float>(0,1)+=2*point.x-2*ni;
//   J.at<float>(0,2)+=2*point.y-2*nj;
  return nearstD; 
}


//update C1+dx ->  C2
void Optimizer::UpdateStateLM(const cv::Mat &dx, const cv::Mat &pose_Old, cv::Mat &pose_New)
{
// 	SkewSymmetricMatrixf dq;
// 	Quaternionf q1, q2;
// 	C1.ToQuaternion(q1.v0123());
  Quaternion q1,q2;
  cv::Vec3f ea1(pose_Old.at<float>(0,0),pose_Old.at<float>(0,1),pose_Old.at<float>(0,2));
  q1.SetEulerAngle(ea1);
  
// 	dq.v012x() = dx.v0123();
// 	//dq X q1 ->q2
// 	Quaternionf::dAxB(dq, q1, q2);
// 	q2.ToRotationMatrix(C2);
  cv::Vec3f dq(dx.at<float>(0,0),dx.at<float>(0,1),dx.at<float>(0,2));
  q2 =q1.dAxB(dq);
  cv::Vec3f ea2=q2.GetEulerAngle();
  

// 	LA::AlignedVector3f t1, t2;
// 	C1.GetTranslation(t1);
// 	t2 = t1;
// 
  cv::Vec3f t1,t2;
  t1= cv::Vec3f(pose_Old.at<float>(0,3),pose_Old.at<float>(0,4),pose_Old.at<float>(0,5));
  t2=t1;
  
// 	SkewSymmetricMatrixf::AddATBTo(dq, t1, t2);
  t2[0] = dq[2] * t1[1] - dq[1] * t1[2] + t2[0];
  t2[1] = dq[0] * t1[2] - dq[2] * t1[0] + t2[1];
  t2[2] = dq[1] * t1[0] - dq[0] * t1[1] + t2[2];
  
  // 	t2.v0() = dx.v3() + t2.v0();
// 	t2.v1() = dx.v4() + t2.v1();
// 	t2.v2() = dx.v5() + t2.v2();
// 	C2.SetTranslation(t2);

  t2[0]+=dx.at<float>(0,3);
  t2[1]+=dx.at<float>(0,4);
  t2[2]+=dx.at<float>(0,5);
  
  for(int i=0;i<3;i++){
    pose_New.at<float>(0,i)=ea2[i];
    pose_New.at<float>(0,i+3)=t2[i];
  }
  
}


