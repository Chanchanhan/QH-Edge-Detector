#include <fstream>      //C++
#include <iostream>
#include<glog/logging.h>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include "ObjectDetector/Optimizer.h"
using namespace OD;
const int MAX_ITERATIN_NUM =20;
const float THREHOLD_ENERGY = 400.0f;
const float STEP_LENGTH = 0.00001f;
const float DX_LENGTH = 0.01f;
const float ENERGY_SIZE = 0.01f;
const float J_SIZE = 0.00001f;

static void printMat(std::string name, cv::Mat M){
//   LOG(INFO)<<name<<std::endl;;
  cout<<name<<endl;
  for(int i=0;i<M.rows;i++){
//     LOG(INFO)<<i<<"th ";
    printf("i %d :",i );
//     char buffer [500000];

    for(int j=0;j<M.cols;j++){
      printf("%f ",M.at<float>(i,j));
//       sprintf(buffer,"%s%d ",buffer,(int)M.at<char>(i,j));
    }

//     LOG(INFO)<<buffer<<std::endl;
    printf("\n");
  }
}
Optimizer::Optimizer(const Config& config, const Mat& initPose, bool is_writer)
{
  
  m_data.m_model->generatePoints();
  
  m_data.m_model = new Model(config);
  //m_data.m_model->LoadGLMModel(config.filename.c_str());
//   m_data.m_correspondence = new Correspondence(config.width,config.height);
//   m_data.m_correspondence->m_lineBundle.create(R,2*L+1,CV_8UC3);
//   m_data.m_correspondence->m_hsv.create(R,2*L+1,CV_8UC3);
  m_calibration = config.camCalibration;
  m_data.m_model->InitPose(initPose);
  m_is_writer = is_writer;
//   if(is_writer)
//   {
//     //record the pose
//     std::string poseFile = "output/out_pose.txt";
//     m_outPose.open(poseFile);
//     if( !m_outPose.is_open() )
//     {
//       printf("Cannot write the pose\n");
//       return;      
//     }
//   }
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
//     printMat("frame",frame);
    std::vector<m_img_point_data> m_img_points_data;  
    for(int i=0;i<frame.rows;i++){
      for(int j=0;j<frame.cols;j++){
	if(frame.at<char>(i,j)>1e0){
	  m_img_points_data.push_back(m_img_point_data(cv::Point(i,j),frame.at<float>(i,j)));
	}      
      }
    }
    std::vector<cv::Point> nPoints;

    cv::Mat _prePose =prePose.clone();
    int itration_num=0;
    cv::Mat newPose=cv::Mat::zeros(1,6,CV_32FC1);
    
    
    m_data.m_model->setVisibleLinesAtPose(_prePose);

    
    float e2 = computeEnergy(_prePose,m_img_points_data,nPoints);

    while(++itration_num<MAX_ITERATIN_NUM){      
      
      
      cv::Mat A= cv::Mat::zeros(6,6,CV_32FC1),b= cv::Mat::zeros(6,1,CV_32FC1);
      if(e2<THREHOLD_ENERGY){
	LOG(WARNING)<<"good init ,no need to optimize!";
	return _prePose;
      }
      constructEnergyFunction(prePose, nPoints,m_img_points_data,A,b);
      Mat A_inverse ;
      cv::invert(A,A_inverse);
    //  printMat("b",b);
    //  printMat("A_inverse",A_inverse);
      Mat dX = -A_inverse*b*STEP_LENGTH;
       printMat("dX",dX);
      
      UpdateStateLM(dX,_prePose,newPose);
      std::vector<cv::Point> new_nPoints;
      float e2_new = computeEnergy(newPose,m_img_points_data,new_nPoints);
      if(e2_new<e2){
	printf("e2 :%f e2_new:%f \n",e2,e2_new);
	computeEnergy(newPose,m_img_points_data,new_nPoints,true);
	_prePose=newPose.clone();
	e2= e2_new;
      }
      if(e2_new<THREHOLD_ENERGY){
	printf("succees optimize!\n");
	return newPose;
      }
      
    }
    LOG(WARNING)<<"to much itration_num!";
    return _prePose;
    
  }
  
  int64 teim0 = cv::getTickCount();
  
  //printf("solving time:%lf\n",(time_s1-time_s)/cv::getTickFrequency());
}

//check move of X in image
// void Optimizer::getMk(cv::Mat dx)
// {
//   
// }


void Optimizer::constructEnergyFunction(const cv::Mat prePose, const std::vector<cv::Point> &nPoints, const std::vector<m_img_point_data>  &m_img_points_data , cv::Mat &A, cv::Mat &b){
  cv::Mat j_X_Pose= cv::Mat::zeros(2,6,CV_32FC1);
  cv::Mat j_Energy_X=cv::Mat::zeros(1,2,CV_32FC1);
  float coffes = 1.0f/nPoints.size();
//   cv::Mat W_to_C= m_data.m_model->GetPoseMatrix(prePose);
  
  
  
  Mat intrinsic = m_data.m_model->getIntrinsic();
  Mat extrinsic = m_data.m_model->GetPoseMatrix(prePose);
  GLMmodel* model =m_data.m_model->GetObjModel();
  cv::Mat pos = m_data.m_model->getPos();
  int size=0;
  for(int i=0;i<model->numLines;++i){
    if(model->lines[i].visible){
      int v0=model->lines[i].vindices[0],v1=model->lines[i].vindices[1];
      Point3f p1= Point3f(pos.at<float>(0,v0-1),pos.at<float>(1,v0-1),pos.at<float>(2,v0-1));
      Point3f p2= Point3f(pos.at<float>(0,v1-1),pos.at<float>(1,v1-1),pos.at<float>(2,v1-1));
      Point3f dx=(p2-p1);
      int Nx=sqrt(dx.x*dx.x+dx.y*dx.y+dx.z*dx.z)/DX_LENGTH;
//       printf(" Nx= %d \n",Nx);

      dx /=Nx;
      size+=Nx;
      Point3f X=p1;
      cv::Mat m_X(4,1,CV_32FC1),m_dX(4,1,CV_32FC1);
      m_X.at<float>(0,0)=X.x,m_X.at<float>(1,0)=X.y,m_X.at<float>(2,0)=X.z,m_X.at<float>(3,0)=1;
      m_dX.at<float>(0,0)=dx.x,m_dX.at<float>(1,0)=dx.y,m_dX.at<float>(2,0)=dx.z,m_dX.at<float>(3,0)=0;

      for(int i=0;i<=Nx;++i,m_X+=m_dX){
	cv::Mat W_X(4,1,CV_32FC1),C_X(4,1,CV_32FC1),P_x(3,1,CV_32FC1);
	cv::Mat _j_X_Pose = cv::Mat::zeros(2,6,CV_32FC1);
	cv::Mat _j_Energy_X=cv::Mat::zeros(1,2,CV_32FC1);
	C_X=extrinsic*m_X;
	P_x=intrinsic*C_X;
	float _x=C_X.at<float>(0,0);
	float _y=C_X.at<float>(0,1);
	float _z=C_X.at<float>(0,2);
	_j_X_Pose.at<float>(0,0)=m_calibration.fx()/_z; 
	_j_X_Pose.at<float>(0,1)=0;
	_j_X_Pose.at<float>(0,2)=-m_calibration.fx()*_x/(_z*_z);
	_j_X_Pose.at<float>(0,3)=-m_calibration.fx()*_x*_y/(_z*_z);
	_j_X_Pose.at<float>(0,4)=m_calibration.fx()*(1+_x*_x/(_z*_z));
	_j_X_Pose.at<float>(0,5)=m_calibration.fx()*_y/_z;

	_j_X_Pose.at<float>(1,0)=0;  
	_j_X_Pose.at<float>(1,1)=m_calibration.fy()/_z;      
	_j_X_Pose.at<float>(1,2)=-m_calibration.fy()*_y/(_z*_z);    
	_j_X_Pose.at<float>(1,3)=-m_calibration.fy()*(1+_y*_y*(1/(_z*_z)));
	_j_X_Pose.at<float>(1,4)=-m_calibration.fy()*_x*_y/(_z*_z);
	_j_X_Pose.at<float>(1,5)=m_calibration.fy()*_x/_z;
    
		
	/*get nearestEdgeDistance ponit*/
	Point point(P_x.at<float>(0,0)/P_x.at<float>(2,0),P_x.at<float>(1,0)/P_x.at<float>(2,0)),nPoint;
	nearestEdgeDistance(point,m_img_points_data,nPoint);
	
//	printf("point: %d %d \n",point.x,point.y);
	_j_Energy_X.at<float>(0,0)=2*(point.x - nPoint.x);
	_j_Energy_X.at<float>(0,1)=2*(point.y - nPoint.y);
	_j_X_Pose*=J_SIZE;
	_j_Energy_X*=J_SIZE;
	
	
// 	_j_X_Pose*=(1.f/Nx);
// 	_j_Energy_X*=(1.f/Nx);
	Mat _J=_j_Energy_X*_j_X_Pose;
	Mat _J_T;
	cv::transpose(_J,_J_T);
	b+=_J_T*( (point.x - nPoint.x)*(point.x - nPoint.x)+(point.y - nPoint.y)*(point.y - nPoint.y));
//     printf("%d : point  %d %d , npoint %d %d\n",i,m_data.m_pointset.m_img_points[i-1].x,m_data.m_pointset.m_img_points[i-1].y,nPoints[i-1].x,nPoints[i-1].y);
	

	j_X_Pose+=_j_X_Pose;
	j_Energy_X+=_j_Energy_X;  
      }

    }

  } 
  Mat J(6,6,CV_32FC1),J_T(6,6,CV_32FC1);
  Mat I = Mat::eye(6,6,CV_32FC1);
  J=j_Energy_X*j_X_Pose *(1.0f/size);  
  printMat("J",J);
  cv::transpose(J,J_T);
  float u=0.5;
//   float norm = A.diag();
  A= J_T*J+I*u;//6*6
 
  printMat("A",A);

  
}

float Optimizer::computeEnergy(const cv::Mat& pose, const std::vector<m_img_point_data> m_img_points_data,std::vector<cv::Point> &nPoints ,const bool printP)
{
  m_data.m_model->GetImagePoints(pose, m_data.m_pointset);
  float energy =0;
  int size =0;
  Mat intrinsic = m_data.m_model->getIntrinsic();
  Mat extrinsic = m_data.m_model->GetPoseMatrix(pose);
  GLMmodel* model =m_data.m_model->GetObjModel();
  cv::Mat pos = m_data.m_model->getPos();
  int lineNum=0;
  for(int i=0;i<model->numLines;++i){
    if(model->lines[i].tovisit){
      lineNum++;
      int v0=model->lines[i].vindices[0],v1=model->lines[i].vindices[1];
      Point3f p1= Point3f(pos.at<float>(0,v0-1),pos.at<float>(1,v0-1),pos.at<float>(2,v0-1));
      Point3f p2= Point3f(pos.at<float>(0,v1-1),pos.at<float>(1,v1-1),pos.at<float>(2,v1-1));
//       printf("P1 : %f %f %f\n",p1.x,p1.y,p1.z);

      Point3f dX=(p2-p1);
      
      int Nx = sqrt(dX.x*dX.x+dX.y*dX.y+dX.z*dX.z)/DX_LENGTH;

//       float Nx=100.f*norm;
      size+=Nx;
      dX /=Nx;
      Point3f X=p1;
      float tmp_energy=0;

      Point2f point1= m_data.m_model->X_to_x(p1,extrinsic);
      Point2f point2= m_data.m_model->X_to_x(p2,extrinsic);

      LOG(INFO)<< "v0:("<<v0<<") "<< point1.x<<" "<<point1.y<< " , v1:("<<v1<<") "<< point2.x<<" "<<point2.y<<std::endl;

      for(int i=0;i<=Nx;++i,X+=dX){
	 cv::Point nPoint;
	 Point2f point= m_data.m_model->X_to_x(X,extrinsic);
	 float e=nearestEdgeDistance(point,m_img_points_data,nPoint);
	 tmp_energy+=nearestEdgeDistance(point,m_img_points_data,nPoint);
	 LOG(INFO)<< i<<"th point :"<<point.x<<" "<<point.y<<"  np:"<<nPoint.x<<" "<<nPoint.y<< "  energy: " <<e<<endl;
	 energy+=e;

	 nPoints.push_back(nPoint);
       }
     
       tmp_energy*=(1.0f/Nx);
       LOG(INFO)<<"mean energy=" <<tmp_energy;

    }

  } 

  energy*=1.0f/size;
  LOG(WARNING)<<" energy =  "<<energy;
  return energy;
  
}

float Optimizer::nearestEdgeDistance(const cv::Point & point,const std::vector<m_img_point_data>  &edge_points,cv::Point &nPoint ,const bool printP)
{
  assert(edge_points.size()>0);  
  float nearstD = -1;
  int ni=0,nj=0;
  
  for(auto edge_point:edge_points){
    float temp_D= (edge_point.m_img_point.x-point.x)*(edge_point.m_img_point.x-point.x)+(edge_point.m_img_point.y-point.y)*(edge_point.m_img_point.y-point.y);
    if(nearstD>temp_D||nearstD==-1){
      nearstD = temp_D;
      nPoint.x =edge_point.m_img_point.x,nPoint.y=edge_point.m_img_point.y;   
      if(printP){
	printf("p %d %d , np %d %d",point.x,point.y,nPoint.x,nPoint.y);
      }
    }
  }
//   J.at<float>(0,1)+=2*point.x-2*ni;
//   J.at<float>(0,2)+=2*point.y-2*nj;
  return ENERGY_SIZE*nearstD; 
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
  printMat("pose_Old",pose_Old);
   printMat("pose_New",pose_New);
}


