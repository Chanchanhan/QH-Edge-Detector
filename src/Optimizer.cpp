#include <fstream>      //C++
#include <iostream>
#include<glog/logging.h>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include<Eigen/Dense>
#include<Eigen/Core>
#include<Eigen/Geometry>
#include"sophus/so3.h"
#include"sophus/se3.h"
#include "POSEST/include/posest_edft.h"
#include "POSEST/include/posest.h"
#include "ObjectDetector/Optimizer.h"
using namespace OD;
const int MAX_ITERATIN_NUM =15;
const float THREHOLD_ENERGY = 170.0f;
const float DX_SIZE = 1.f;
const float NX_LENGTH = 0.02f;
const float ENERGY_SIZE = 1.0f;
const float J_SIZE = 0.00001f;
const float LM_STEP =10 ;
const float INIT_LAMDA =1;
const float SIZE_A =1;
const float INF =1e10;
const float THREHOLD_DX= 1.0e17;
const bool  USE_SOPHUS = 0;
const bool  USE_MY_TRANSFORMATION = 1;
const float MAX_VALIAD_DISTANCE = 100.f;
#define FACTOR_DEG_TO_RAD 0.01745329252222222222222222222222f
#define PRIOR_MAX_DEVIATION_CAMERA_HEIGHT			0.5f
#define PRIOR_MAX_DEVIATION_CAMERA_ROTATION_XY			5.0f
#define PRIOR_MAX_DEVIATION_CAMERA_ROTATION_Z			30.0f
#define SIMULATION_MAX_ROTATIONS				3
#define SIMULATION_MAX_ROTATION_RATIO			0.3f
#define SIMULATION_MAX_TRANSLATIONS				1
#define SIMULATION_MAX_TRANSLATION_RATIO		0.5f
#define USE_PNP
// #define EDF_TRAKER
//  #define DRAW_LINE_P2NP
typedef Eigen::Matrix<double,6,1> Vector6d;

struct EDFTdata 
{
	double *K; // intrinsics
	float *distMap; // distance map
	double(*ctrPts3D)[3]; // contour points
	int nCtrPts; // the number of contour points

	int width, height; // the (width, height) of distMap
};


Optimizer::Optimizer(const Config& config, const float * initPose, bool is_writer):imgHeight( config.height), imgWidth(config.width),  m_calibration ( config.camCalibration),m_is_writer(is_writer)
{
  m_data.m_model = new Model(config);
  dist = (float *)malloc(imgHeight* imgWidth * sizeof(float)); 
  final_e = 1E9;

}

Optimizer::~Optimizer()
{
  if(m_is_writer)
  {
    m_outPose.close();    
  }
}


void Optimizer::optimizingLM(const float * prePose,const cv::Mat& curFrame,const cv::Mat& distFrame, const cv::Mat &locations, const int frameId,float * _newPose )
{
   int64 time0 = cv::getTickCount();
  {
    mFrame=curFrame;
    LOG(WARNING)<<" ";
    LOG(WARNING)<<"frameId = "<<frameId;
    _locations=(int *)locations.data;
    imgHeight=distFrame.cols;
    imgWidth=distFrame.rows;
    cv::Mat A_I=cv::Mat::eye(6,6,CV_32FC1);
    float lamda=INIT_LAMDA;
    std::vector<cv::Point> nPoints;
    m_Transformation.setPose(prePose);
    int itration_num=0;
    Transformation newTransformation;
    m_data.m_model->setVisibleLinesAtPose(m_Transformation.Pose());    
    float e2 = computeEnergy(distFrame, m_Transformation.Pose());
    if(e2<THREHOLD_ENERGY){
	LOG(INFO)<<"good init ,no need to optimize! energy = "<<e2;
	return ;
      }else{      	
	LOG(WARNING)<<"to optimize with energy = "<<e2;	
    }
    while(++itration_num<MAX_ITERATIN_NUM){    
      Sophus::SE3 T_SE3;	      
      LOG(INFO)<<"a itration_num = " <<itration_num;
      cv::Mat _A= cv::Mat::zeros(6,6,CV_32FC1),b= cv::Mat::zeros(6,1,CV_32FC1),A= cv::Mat::zeros(6,6,CV_32FC1);      
      Mat A_inverse,dX ;
      float coarsePose[6]={0};
//       coarsePose=m_Transformation.M_Pose().clone();
#ifdef USE_PNP
       getCoarsePoseByPNP(m_Transformation.Pose(),distFrame,coarsePose);
      LOG(INFO)<<"pre pose"<<m_Transformation.M_Pose();
      m_Transformation.setPose(coarsePose);
      LOG(INFO)<<"coarse pose"<<m_Transformation.M_Pose();
#endif
      constructEnergyFunction(distFrame,m_Transformation.Pose(),A_I,lamda, _A,b);
      _A/=abs(_A.at<float>(0,0));
      A=_A+A_I*lamda;
      cv::invert(A,A_inverse);
      //get dX
      {
	LOG(INFO)<<"A_I\n"<<A_I;
      LOG(INFO)<<"lamda = "<<lamda;
      LOG(INFO)<<"A\n"<<A;
      b/=abs(b.at<float>(0,0));
      LOG(INFO)<<"A_inverse\n"<<A_inverse;      
      LOG(INFO)<<"b\n"<<b;
      dX =- A_inverse*b*DX_SIZE;
      LOG(WARNING)<<"dX "<<dX;
      }
  

      float e2_new;
      if(USE_MY_TRANSFORMATION){
	newTransformation.setPose(m_Transformation.Pose());
	newTransformation.xTransformation(dX);
	e2_new = computeEnergy(distFrame, newTransformation.Pose());
      }
      else if(USE_SOPHUS)
      {      
	Eigen::Vector3d v3d(m_Transformation.u1(),m_Transformation.u2(),m_Transformation.u3());     
	Eigen::Vector3d translationV3(m_Transformation.x(),m_Transformation.y(),m_Transformation.z());     
	T_SE3 =Sophus::SE3( Sophus::SO3::exp(v3d),translationV3); 
	Vector6d se3_Update;
	for(int i=0;i<3;i++){
	  se3_Update(i,0)=(double)dX.at<float>(0,i+3);
	  se3_Update(i+3,0)=(double)dX.at<float>(0,i);
	}
	Sophus::SE3 update_SE3=Sophus::SE3::exp(se3_Update); 
	Sophus::SE3 new_T_SE3=Sophus::SE3::exp(se3_Update)*T_SE3;	
	newTransformation.setPoseFromTransformationMatrix(new_T_SE3.matrix());
	e2_new = computeEnergy(distFrame, newTransformation.Pose());

      }/*else{
	  UpdateStateLM(dX,m_Transformation.M_Pose(),newPose);
	  newTransformation.setPose(newPose);
	  e2_new = computeEnergy(distFrame, newPose);
      }*/
#ifndef EDF_TRAKER
      while(e2_new>e2){	 
	  LOG(WARNING)<<"sorry!!!Not to optimize! e2 :"<<e2<<" e2_new: "<<e2_new<<" \n";
	  lamda*=LM_STEP;
	  A=_A+A_I*lamda;
	  cv::invert(A,A_inverse);
	  
	  b/=abs(b.at<float>(0,0));
// 	  LOG(INFO)<<"A_inverse\n"<<A_inverse;      
// 	  LOG(INFO)<<"b\n"<<b;
	  Mat dX =- A_inverse*b*DX_SIZE;
	  LOG(WARNING)<<"dX "<<dX;
	  float lastE2;
	  
	  /*
 	  e2_new = computeEnergy(frame, newTransformation.M_Pose());
 	  LOG(WARNING)<<"newTransformation.M_Pose(): "<<newTransformation.M_Pose()<<" e2_new(newTransformation) = "<<e2_new;*/
	  if(USE_MY_TRANSFORMATION){
	    newTransformation.setPose(m_Transformation.Pose(),true);
	    newTransformation.xTransformation(dX);
	    e2_new = computeEnergy(distFrame, newTransformation.Pose());
	  }
	  else if(USE_SOPHUS)
	  {
	    Vector6d se3_Update;
	    for(int i=0;i<3;i++){
	      se3_Update(i,0)=(double)dX.at<float>(0,i+3);
	      se3_Update(i+3,0)=(double)dX.at<float>(0,i);
	    }
	    Sophus::SE3 update_SE3=Sophus::SE3::exp(se3_Update); 
	    Sophus::SE3 new_T_SE3=Sophus::SE3::exp(se3_Update)*T_SE3;	
	    newTransformation.setPoseFromTransformationMatrix(new_T_SE3.matrix());
	    e2_new = computeEnergy(distFrame, newTransformation.Pose());
	  }/*else{
	    UpdateStateLM(dX,m_Transformation.M_Pose(),newPose);	
	    newTransformation.setPose(newPose);
	    e2_new = computeEnergy(distFrame, newPose);
	  }*/
	  	  
	 
	  lastE2=e2_new;

	  LOG(WARNING)<<"newPose"<<newTransformation.M_Pose()<<"  e2_new(newPose) = "<<e2_new;

	  itration_num++;
	  if(itration_num>MAX_ITERATIN_NUM){
	    LOG(INFO)<<"to much itration_num!";
	    memcpy(_newPose,m_Transformation.Pose(),sizeof(float)*6);

	    return /*m_Transformation.M_Pose()*/;    
	  }
	  if(fabs(lastE2-e2_new)<1e-5){
	    memcpy(_newPose,m_Transformation.Pose(),sizeof(float)*6);
	    continue ;  
	  }
	
      }
#else
  int cstfunc = POSEST_EDFT_ERR_NLN;
  float newPose[6];
  int ret=edfTracker(m_Transformation.Pose(), distFrame,cstfunc,newPose);
  newTransformation.setPose(newPose);
  e2_new = computeEnergy(distFrame, newTransformation.Pose());
  LOG(WARNING)<<"ret = "<<ret<<" Pose : "<<newTransformation.M_Pose()<<" energy = "<<e2_new;
#endif
    	     
      if(e2_new>=e2){
#ifndef EDF_TRAKER


	lamda*=LM_STEP;
	continue;
#else
	
#endif
      }else{      
	LOG(WARNING)<<"good !To optimize! e2 :"<<e2<<" e2_new: "<<e2_new;
	LOG(INFO)<<"dX "<<dX;
	LOG(INFO)<<"newPose "<<newTransformation.M_Pose();
	A_I=A.clone();
	A_I/=abs(A_I.at<float>(0,0));
	m_Transformation.setPose(newTransformation.Pose());         
//         m_Transformation.setPose(newTransformation.M_Pose());

	e2= e2_new;
	lamda=INIT_LAMDA;
      }
      if(e2_new<THREHOLD_ENERGY){
	LOG(WARNING)<<"succees optimize!";
	memcpy(_newPose,m_Transformation.Pose(),sizeof(float)*6);

	return ;
	
      }
     
    }
    LOG(INFO)<<"to much itration_num!";
  }
  
  int64 teim0 = cv::getTickCount();
}



cv::Point Optimizer::getNearstPointLocation(const cv::Point &point){
  int x= point.y;
  int y= point.x;   			
  while(_locations[y + imgHeight * x]!=x||_locations[imgWidth*imgHeight+y + imgHeight * x]!=y){
    x=_locations[y + imgHeight * x];
    y=_locations[imgWidth*imgHeight+y + imgHeight* x];
//      cout<<"move to: "<<x<<" y: "<<y<<endl;
   }
//    cout<<"End to: x "<<x<<" y: "<<y<<endl;
   return Point(y,x);
}
void Optimizer::constructEnergyFunction(const cv::Mat distFrame,const float* prePose,const cv::Mat &lastA,const int &lamda, cv::Mat &A, cv::Mat &b){
 cv::Mat j_X_Pose= cv::Mat::zeros(2,6,CV_32FC1);
  cv::Mat j_Energy_X=cv::Mat::zeros(1,2,CV_32FC1);
  
  Mat intrinsic = m_data.m_model->getIntrinsic();
  Mat extrinsic = Transformation::getTransformationMatrix(prePose);
  GLMmodel* model =m_data.m_model->GetObjModel();
  cv::Mat pos = m_data.m_model->getPos();
  int size=0;
#ifdef DRAW_LINE_P2NP
  cv::Mat drawFrame/* = mFrame.clone()*/;
  cv::cvtColor(distFrame/255.f, drawFrame, CV_GRAY2BGR);
#endif
  for(int i=0;i<model->numLines;++i){
    if(model->lines[i].tovisit){
      int v0=model->lines[i].vindices[0],v1=model->lines[i].vindices[1];
      Point3f p1= Point3f(pos.at<float>(0,v0-1),pos.at<float>(1,v0-1),pos.at<float>(2,v0-1));
      Point3f p2= Point3f(pos.at<float>(0,v1-1),pos.at<float>(1,v1-1),pos.at<float>(2,v1-1));
      Point3f dx=(p2-p1);
      int Nx=sqrt(dx.x*dx.x+dx.y*dx.y+dx.z*dx.z)/NX_LENGTH;
//       printf(" Nx= %d \n",Nx);
      Point point1= m_data.m_model->X_to_x(p1,extrinsic);
      Point point2= m_data.m_model->X_to_x(p2,extrinsic);
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
		
	/*get nearestEdgeDistance ponit*/
	Point point(P_x.at<float>(0,0)/P_x.at<float>(2,0),P_x.at<float>(1,0)/P_x.at<float>(2,0));
	Point nearstPoint = getNearstPointLocation(point);
		
	float dist2Edge=getDistanceToEdege(point1,point2,nearstPoint);

	if(distFrame.at<float>(nearstPoint)>=254||dist2Edge>MAX_VALIAD_DISTANCE){
	  continue;
	}
#ifdef DRAW_LINE_P2NP	

	m_data.m_model->DisplayLine(point,nearstPoint,drawFrame,sqrt(dist2Edge));
 #endif
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
		

	
// 	_j_Energy_X.at<float>(0,0)=2*((point1.y-point2.y)*nearstPoint.x +
// 				     ((point1.x*point2.y-point2.x*point1.y)+(point2.x-point1.y)*nearstPoint.y)*(point1.y-point2.y))
// 				    * (1.f/(pow((point2.x-point1.x),2)+pow((point2.y-point1.y),2)));
// 
// 
// 	_j_Energy_X.at<float>(0,1)=2*((point2.x-point1.x)*nearstPoint.y +
// 				     ((point1.x*point2.y-point2.x*point1.y)+(point1.x-point2.y)*nearstPoint.x)*(point2.x-point1.x))
// 				    * (1.f/(pow((point2.x-point1.x),2)+pow((point2.y-point1.y),2)));
	

	_j_Energy_X.at<float>(0,0)=2*(point.x - nearstPoint.x);
	_j_Energy_X.at<float>(0,1)=2*(point.y - nearstPoint.y);
	_j_X_Pose*=J_SIZE;
	_j_Energy_X*=J_SIZE;
	
// 	_j_X_Pose*=(1.f/Nx);
// 	_j_Energy_X*=(1.f/Nx);
	Mat _J=_j_Energy_X*_j_X_Pose;
	Mat _J_T;
	cv::transpose(_J,_J_T);
// 	b+=_J_T*(sqrt(getDistanceToEdege(point1,point2,point)));
	b+=_J_T*( (point.x - nearstPoint.x)*(point.x - nearstPoint.x)+(point.y - nearstPoint.y)*(point.y - nearstPoint.y));
//     printf("%d : point  %d %d , npoint %d %d\n",i,m_data.m_pointset.m_img_points[i-1].x,m_data.m_pointset.m_img_points[i-1].y,nPoints[i-1].x,nPoints[i-1].y);
	

	j_X_Pose+=_j_X_Pose;
	j_Energy_X+=_j_Energy_X;  
      }

    }

  } 
  /*
  LOG(WARNING)<<"_j_X_Pose\n"<<j_X_Pose;
  LOG(WARNING)<<"_j_Energy_X\n"<<j_Energy_X;*/
  
#ifdef DRAW_LINE_P2NP
LOG(WARNING)<<"to draw drawFrame";
  m_data.m_model->DisplayCV(prePose,drawFrame);
  imshow("DRAW_LINE_P2NP",drawFrame);
//   imshow("distMap",frame/255.f);
  waitKey(0);
#endif
  
  Mat J(6,6,CV_32FC1),J_T(6,6,CV_32FC1);
  J=j_Energy_X*j_X_Pose *(1.0f/size)*(1.0f/size);  
//   printMat("J",J);
  cv::transpose(J,J_T);
//   float norm = A.diag();
  A= J_T*J+lastA*lamda;//6*6
  A*=SIZE_A;
//   printMat("A",A);

  
}

float Optimizer::getDistanceToEdege(const Point& e0, const Point& e1, const Point& v)
{
  return pow((e0.y-e1.y)*v.x +(e1.x-e0.x)*v.y+(e0.x*e1.y-e1.x*e0.y),2)/
	  (pow((e1.x-e0.x),2)+pow((e1.y-e0.y),2));
}
int Optimizer::edfTracker(const float* prePose, const Mat& distMap,const  int NLrefine, float* newPose)
{
  Mat _distMap=distMap.clone();
  if (_distMap.isContinuous()/* && distMap.depth()==CV_32FC1*/)
  {
    memcpy(dist, _distMap.data, imgHeight * imgWidth * sizeof(float));    
  }
  else
  {
    for (int i = 0; i < imgHeight; ++i)
    {
      float *rptr =_distMap.ptr<float>(i);
      for (int j = 0; j < imgWidth; ++j)
      {
	dist[i*imgWidth + j] = rptr[j];	
      }      
    }    
  }
  
  
  
  m_data.m_model->GetImagePoints(prePose, m_data.m_pointset);
  float energy =0;
  int size =0;
  Mat intrinsic = m_data.m_model->getIntrinsic();
  Mat extrinsic = m_data.m_model->GetPoseMatrix(prePose);
  GLMmodel* model =m_data.m_model->GetObjModel();
  cv::Mat pos = m_data.m_model->getPos();
  int lineNum=0;       
  float meanDX=0;
  std::vector<cv::Point3d> contourPoints ;
  for(int i=0;i<model->numLines;++i){
    if(model->lines[i].tovisit){
      
      lineNum++;
      int v0=model->lines[i].vindices[0],v1=model->lines[i].vindices[1];
      Point3f p1= Point3f(pos.at<float>(0,v0-1),pos.at<float>(1,v0-1),pos.at<float>(2,v0-1));
      Point3f p2= Point3f(pos.at<float>(0,v1-1),pos.at<float>(1,v1-1),pos.at<float>(2,v1-1));
      Point3f dX=(p2-p1);      
      int Nx = sqrt(dX.x*dX.x+dX.y*dX.y+dX.z*dX.z)/NX_LENGTH;
      dX /=Nx;
      Point3f X=p1;
      Point point1= m_data.m_model->X_to_x(p1,extrinsic);
      Point point2= m_data.m_model->X_to_x(p2,extrinsic);
      if(point1.x<0||point1.y<0||point2.x<0||point2.y<0||point1.x>=imgWidth||point2.x>=imgWidth||point1.y>=imgHeight||point2.y>=imgHeight){
	continue;
      }
      float meanE_LINE=0;
      for(int i=0;i<=Nx;++i,X+=dX){
	 contourPoints.push_back(Point3d(X.x,X.y,X.z));
      }  
    }
  }
  double(*ctrPts3D)[3];
  int nCtrPts=contourPoints.size();
  ctrPts3D = (double(*)[3])malloc(nCtrPts*sizeof(double[3]));
  for (int i = 0; i < nCtrPts; ++i)
  {
    ctrPts3D[i][0] = contourPoints[i].x;
    ctrPts3D[i][1] = contourPoints[i].y;
    ctrPts3D[i][2] = contourPoints[i].z;    
  }
  double K[9] = { m_calibration.fx(), 0, m_calibration.cx(), 0, m_calibration.fy(), m_calibration.cy(), 0, 0, 1 };
  double _newPose[6];
  int ret = posest_edft(dist, ctrPts3D, nCtrPts,
		imgWidth, imgHeight, K, _newPose, 6, NLrefine, 1, &final_e);
  for(int i=0;i<6;i++){
    newPose[i]=_newPose[i];
  }
  return ret;
}

void Optimizer::getCoarsePoseByPNP(const float *prePose, const Mat &distMap,float *coarsePose)
{
  m_data.m_model->GetImagePoints(prePose, m_data.m_pointset);
  float energy =0;
  int size =0;
  Mat intrinsic = m_data.m_model->getIntrinsic();
  Mat extrinsic = m_data.m_model->GetPoseMatrix(prePose);
  GLMmodel* model =m_data.m_model->GetObjModel();
  cv::Mat pos = m_data.m_model->getPos();
  int lineNum=0;       
  float meanDX=0;
  std::vector<cv::Point2d> imagePoints ;
  std::vector<cv::Point3d> objectPoints ;
  for(int i=0;i<model->numLines;++i){
    if(model->lines[i].tovisit){
      
      lineNum++;
      int v0=model->lines[i].vindices[0],v1=model->lines[i].vindices[1];
      Point3f p1= Point3f(pos.at<float>(0,v0-1),pos.at<float>(1,v0-1),pos.at<float>(2,v0-1));
      Point3f p2= Point3f(pos.at<float>(0,v1-1),pos.at<float>(1,v1-1),pos.at<float>(2,v1-1));

      Point3f dX=(p2-p1);      
      int Nx = sqrt(dX.x*dX.x+dX.y*dX.y+dX.z*dX.z)/NX_LENGTH;

      size+=Nx;
      dX /=Nx;
      Point3f X=p1;

      Point point1= m_data.m_model->X_to_x(p1,extrinsic);
      Point point2= m_data.m_model->X_to_x(p2,extrinsic);
//       LOG(INFO)<< "v0:("<<v0<<") "<< point1.x<<" "<<point1.y<< " , v1:("<<v1<<") "<< point2.x<<" "<<point2.y<<std::endl;
      if(point1.x<0||point1.y<0||point2.x<0||point2.y<0||point1.x>=distMap.size().width||point2.x>=distMap.size().width||point1.y>=distMap.size().height||point2.y>=distMap.size().height){
	continue;
      }
      float meanE_LINE=0;
      for(int i=0;i<=Nx;++i,X+=dX){
	 Point point= m_data.m_model->X_to_x(X,extrinsic);	 	 
	 Point nearst=getNearstPointLocation(point);
	 objectPoints.push_back(Point3d(X.x,X.y,X.z));
	 imagePoints.push_back(Point2d(nearst.x,nearst.y));
      }  
    }
  }
  cv::Mat distCoeffs(4,1,cv::DataType<double>::type);
  distCoeffs.at<double>(0) = 0;
  distCoeffs.at<double>(1) = 0;
  distCoeffs.at<double>(2) = 0;
  distCoeffs.at<double>(3) = 0;
  cv::Mat rvec(3,1,cv::DataType<double>::type);
  cv::Mat tvec(3,1,cv::DataType<double>::type);
      
   LOG(INFO)<<"coarsePose in: "<<prePose[0]<<" "<<prePose[1]<<" "<<prePose[2]<<" "<<prePose[3]<<" "<<prePose[4]<<" "<<prePose[5]<<" ";

   for(int i=0;i<3;i++){
     rvec.at<double>(i,0)=(double)prePose[i];
     tvec.at<double>(i,0)=(double)prePose[i+3];
   }

  cv::Mat cameraMatrix(3,3,cv::DataType<double>::type);
  cameraMatrix.at<double>(0,0)=m_calibration.fx();
  cameraMatrix.at<double>(1,1)=m_calibration.fy();
  cameraMatrix.at<double>(0,2)=m_calibration.cx();
  cameraMatrix.at<double>(1,2)=m_calibration.cy();
  cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec,true,cv::SOLVEPNP_EPNP);

 for(int i=0;i<3;++i){
    coarsePose[i]=(float)rvec.at<double>(i,0);
    coarsePose[i+3]=(float)tvec.at<double>(i,0);
  }
   LOG(INFO)<<"coarsePose out: "<<coarsePose[0]<<" "<<coarsePose[1]<<" "<<coarsePose[2]<<" "<<coarsePose[3]<<" "<<coarsePose[4]<<" "<<coarsePose[5]<<" ";

}

float Optimizer::computeEnergy(const cv::Mat& frame,const float * pose)
{
 m_data.m_model->GetImagePoints(pose, m_data.m_pointset);
  float energy =0;
  int size =0;
  Mat intrinsic = m_data.m_model->getIntrinsic();
  Mat extrinsic = Transformation::getTransformationMatrix(pose);
  GLMmodel* model =m_data.m_model->GetObjModel();
  cv::Mat pos = m_data.m_model->getPos();
  int lineNum=0;       
  float meanDX=0;
  
  for(int i=0;i<model->numLines;++i){
    if(model->lines[i].tovisit){
      lineNum++;
      int v0=model->lines[i].vindices[0],v1=model->lines[i].vindices[1];
      Point3f p1= Point3f(pos.at<float>(0,v0-1),pos.at<float>(1,v0-1),pos.at<float>(2,v0-1));
      Point3f p2= Point3f(pos.at<float>(0,v1-1),pos.at<float>(1,v1-1),pos.at<float>(2,v1-1));
//       printf("P1 : %f %f %f\n",p1.x,p1.y,p1.z);

      Point3f dX=(p2-p1);      
      int Nx = sqrt(dX.x*dX.x+dX.y*dX.y+dX.z*dX.z)/NX_LENGTH;

//       float Nx=100.f*norm;
      size+=Nx;
      dX /=Nx;
      Point3f X=p1;

      Point point1= m_data.m_model->X_to_x(p1,extrinsic);
      Point point2= m_data.m_model->X_to_x(p2,extrinsic);

       
      LOG(INFO)<< "v0:("<<v0<<") "<< point1.x<<" "<<point1.y<< " , v1:("<<v1<<") "<< point2.x<<" "<<point2.y<<std::endl;
      if(point1.x<0||point1.y<0||point2.x<0||point2.y<0||point1.x>=frame.size().width||point2.x>=frame.size().width||point1.y>=frame.size().height||point2.y>=frame.size().height){
	return INF;
      }
      float meanE_LINE=0;

       for(int i=0;i<=Nx;++i,X+=dX){
	 Point point= m_data.m_model->X_to_x(X,extrinsic);	 	 
	 Point nearst=getNearstPointLocation(point);
	 
	 float de2 = /*getDistance2ToEdege(point1,point2,nearst)+*/frame.at<float>(point);
	 float dist2Edge=getDistanceToEdege(point1,point2,nearst);
	 //enlarge influence of 255
	 if(de2==255.f){
	    de2*=10;
	 }
	 
	 LOG(INFO)<< i<<"th point :"<<point.x<<" "<<point.y<< "  energy: " << frame.at<float>(point)<<" Distance energy: "<<de2<<"  np: "<<nearst.x<<" "<<nearst.y<<" dist2Edge ="<< dist2Edge;	 
//	 if(DX<THREHOLD_DX||DX<meanE_LINE*meanE_LINE){
	  energy+=de2;
//	 }
  
       }

    }
  } 

	
  energy*=1.0f/size;
  LOG(INFO)<<"Total mean Energy = "<<energy;

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
  return ENERGY_SIZE*nearstD; 
}
void Optimizer::UpdateStateLM(const cv::Mat &dx, const float * pose_Old, Transformation &transformation_New)
{
  
  transformation_New.setPose(pose_Old);  
   LOG(INFO)<<"pose_Old"<<transformation_New.Pose();
  transformation_New.xTransformation(dx);
   LOG(INFO)<<"transformation_New"<<transformation_New.transformationMatrix();

}

//update C1+dx ->  C2
void Optimizer::UpdateStateLM(const cv::Mat &dx, const float * pose_Old, float * pose_New)
{
// 	SkewSymmetricMatrixf dq;
// 	Quaternionf q1, q2;
// 	C1.ToQuaternion(q1.v0123());
  Quaternion q1,q2;
  cv::Vec3f rotV(pose_Old[0],pose_Old[1],pose_Old[2]);
  q1.SetRotVec(rotV);
  
// 	dq.v012x() = dx.v0123();
// 	//dq X q1 ->q2
// 	Quaternionf::dAxB(dq, q1, q2);
// 	q2.ToRotationMatrix(C2);
  cv::Vec3f dq(dx.at<float>(0,0),dx.at<float>(0,1),dx.at<float>(0,2));
  q2 =q1.dAxB(dq);
  
  
  cv::Vec3f rotV_New=q2.GetRotVec();
//    cv::Vec3f ea2=q2.GetEulerAngle();
  

// 	LA::AlignedVector3f t1, t2;
// 	C1.GetTranslation(t1);
// 	t2 = t1;
// 
  cv::Vec3f t1,t2;
  t1= cv::Vec3f(pose_Old[3],pose_Old[4],pose_Old[5]);
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
    pose_New[i]=rotV_New[i];
    pose_New[i+3]=t2[i];
  }
//   LOG(INFO)<<"pose_Old "<<pose_Old;
//   LOG(INFO)<<"pose_New "<<pose_New;
}


