
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
#include "Traker/Traker.h"
#include "Image/ImgProcession.h"

const float INF =1e10;

typedef Eigen::Matrix<double,6,1> Vector6d;

struct EDFTdata 
{
	double *K; // intrinsics
	float *distMap; // distance map
	double(*ctrPts3D)[3]; // contour points
	int nCtrPts; // the number of contour points

	int width, height; // the (width, height) of distMap
};

using namespace OD;

namespace OD{
  
Traker::Traker(const float * initPose,const bool is_writer):imgHeight( Config::configInstance().VIDEO_HEIGHT), imgWidth(Config::configInstance().VIDEO_WIDTH),  m_calibration ( Config::configInstance().camCalibration),m_is_writer(is_writer)
{
  m_data.m_model = new Model(Config::configInstance());
  dist = (float *)malloc(imgHeight* imgWidth * sizeof(float)); 
  final_e = 1E9;
  m_frameId=-1;
}

Traker::~Traker()
{
  if(m_is_writer)
  {
    m_outPose.close();    
  }
  free(_locations);
  free(dist);
  free(mFrame.data);
  free(locationsMat.data);
  free(distFrame.data);
  
}


int Traker::toTrack(const float * prePose,const cv::Mat& curFrame,const int & frameId,float * _newPose,float &finalE2 )
{
  
   int64 time0 = cv::getTickCount();
  {
    if(m_frameId!=frameId){
      m_frameId=frameId;
      getDistMap(curFrame);
      _locations=(int *)locationsMat.data;
    }
    
    mFrame=curFrame;
    LOG(WARNING)<<" ";
    LOG(WARNING)<<"frameId = "<<frameId;
    
//     imgHeight=distFrame.cols;
//     imgWidth=distFrame.rows;
    cv::Mat A_I=cv::Mat::eye(6,6,CV_32FC1);
    float lamda=Config::configInstance().INIT_LAMDA;
    std::vector<cv::Point> nPoints;
    m_Transformation.setPose(prePose);
    int itration_num=0;
    Transformation newTransformation;
    m_data.m_model->setVisibleLinesAtPose(m_Transformation.Pose());    
    float e2 = computeEnergy(distFrame, m_Transformation.Pose());
    if(e2<Config::configInstance().OPTIMIZER_THREHOLD_ENERGY){
	LOG(INFO)<<"good init ,no need to optimize! energy = "<<e2;
	finalE2 =e2;
	return 1;
      }else{      
	
	LOG(WARNING)<<"to optimize with energy = "<<e2<<"m_Transformation : "<<m_Transformation.M_Pose()<<" Config::configInstance().OPTIMIZER_THREHOLD_ENERGY = "<<Config::configInstance().OPTIMIZER_THREHOLD_ENERGY;	
    }
    while(++itration_num<Config::configInstance().OPTIMIZER_MAX_ITERATIN_NUM){    
      Sophus::SE3 T_SE3;	      
      LOG(INFO)<<"a itration_num = " <<itration_num;
      cv::Mat _A= cv::Mat::zeros(6,6,CV_32FC1),b= cv::Mat::zeros(6,1,CV_32FC1),A= cv::Mat::zeros(6,6,CV_32FC1);      
      Mat A_inverse,dX ;
      if(Config::configInstance().USE_PNP){
      float coarsePose[6]={0};
	getCoarsePoseByPNP(m_Transformation.Pose(),distFrame,coarsePose);
	LOG(INFO)<<"pre pose"<<m_Transformation.M_Pose();
//      if(computeEnergy(distFrame,coarsePose)<computeEnergy(distFrame,m_Transformation.Pose())){
	  m_Transformation.setPose(coarsePose,true);	  
// 	 LOG(INFO)<<" update to coarse pose"<<m_Transformation.M_Pose();

      }

      constructEnergyFunction2(distFrame,m_Transformation.Pose(),A_I,lamda, _A,b);
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
	dX =- A_inverse*b*Config::configInstance().DX_SIZE;
	LOG(WARNING)<<"dX "<<dX;
      }
  

      float e2_new;
      updateState(distFrame,dX,m_Transformation,newTransformation,e2_new);
	    memcpy(_newPose,m_Transformation.Pose(),sizeof(float)*6);

#ifndef EDF_TRAKER
      while(e2_new>=e2){	 	
	  if(itration_num>Config::configInstance().OPTIMIZER_MAX_ITERATIN_NUM){
	    LOG(INFO)<<"to much itration_num!";
	    memcpy(_newPose,m_Transformation.Pose(),sizeof(float)*6);
	    finalE2= computeEnergy(distFrame, _newPose);
	    return -1;    
	  }
	  
	  LOG(WARNING)<<"sorry!!!Not to optimize! e2 :"<<e2<<" e2_new: "<<e2_new<<" \n";

	  lamda*=Config::configInstance().LM_STEP;
	  A=_A+A_I*lamda;
	  cv::invert(A,A_inverse);	  
	  b/=abs(b.at<float>(0,0));
// 	  LOG(INFO)<<"A_inverse\n"<<A_inverse;      
// 	  LOG(INFO)<<"b\n"<<b;
	  Mat dX =- A_inverse*b*Config::configInstance().DX_SIZE;
	  LOG(WARNING)<<"dX "<<dX;
	  float lastE2=e2_new;
	  updateState(distFrame,dX,m_Transformation,newTransformation,e2_new);	  	  
		  	  
	  LOG(WARNING)<<"newPose"<<newTransformation.M_Pose()<<"  e2_new(newPose) = "<<e2_new;
	  itration_num++;
	  
	  if(fabs(lastE2==e2_new)<1e-10){
	    LOG(WARNING)<<"Stop to LM-iteration ,beacuse lastE2>>E2";
// 	    break;
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


	lamda*=Config::configInstance().LM_STEP;
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
	lamda=Config::configInstance().INIT_LAMDA;
      }
      if(e2_new<Config::configInstance().OPTIMIZER_THREHOLD_ENERGY){
	LOG(WARNING)<<"succees optimize!";
	memcpy(_newPose,m_Transformation.Pose(),sizeof(float)*6);
	finalE2 =computeEnergy(distFrame, _newPose);

	return 1;
	
      }
     
    }

        memcpy(_newPose,prePose,sizeof(float)*6);
    finalE2= computeEnergy(distFrame, _newPose);
    LOG(INFO)<<"to much itration_num!";
    return -1;
  }
  
  int64 teim0 = cv::getTickCount();
}



cv::Point Traker::getNearstPointLocation(const cv::Point &point){
  int x= point.y;
  int y= point.x;   			
  while(_locations[y + imgWidth * x]!=x||_locations[imgHeight *imgWidth+y + imgWidth * x]!=y){
    x=_locations[y + imgWidth * x];
    y=_locations[imgHeight *imgWidth+y + imgWidth* x];
//      cout<<"move to: "<<x<<" y: "<<y<<endl;
   }
//    cout<<"End to: x "<<x<<" y: "<<y<<endl;
   return Point(y,x);
}
void Traker::constructEnergyFunction2(const cv::Mat distFrame,const float* prePose,const cv::Mat &lastA,const int &lamda, cv::Mat &A, cv::Mat &b)
{
  
  Mat intrinsic = m_data.m_model->getIntrinsic();
  Mat extrinsic = Transformation::getTransformationMatrix(prePose);
  GLMmodel* model =m_data.m_model->GetObjModel();
  cv::Mat pos = m_data.m_model->getPos();
  cv::Mat drawFrame;
  if(Config::configInstance().CV_LINE_P2NP){    
    cv::cvtColor(distFrame/255.f, drawFrame, CV_GRAY2BGR);
  }
  
        
        
  cv::Mat j_X_Pose= cv::Mat::zeros(2,6,CV_32FC1);
  cv::Mat j_Energy_X=cv::Mat::zeros(1,2,CV_32FC1);  
  cv::Mat mat_b(6, N_Points, CV_32FC1);   
  cv::Mat J_X_Pose = cv::Mat::zeros(2*N_Points,6,CV_32FC1);
  cv::Mat J_Energy_X=cv::Mat::zeros(1,2*N_Points,CV_32FC1);
  for(int i=0;i<model->numLines;++i){
    if(model->lines[i].tovisit){
      int v0=model->lines[i].vindices[0],v1=model->lines[i].vindices[1];
      Point3f p1= Point3f(pos.at<float>(0,v0-1),pos.at<float>(1,v0-1),pos.at<float>(2,v0-1));
      Point3f p2= Point3f(pos.at<float>(0,v1-1),pos.at<float>(1,v1-1),pos.at<float>(2,v1-1));
      Point3f dx=(p2-p1);
      int Nx=sqrt(dx.x*dx.x+dx.y*dx.y+dx.z*dx.z)/Config::configInstance().NX_LENGTH;
//       printf(" Nx= %d \n",Nx);
      Point point1= m_data.m_model->X_to_x(p1,extrinsic);
      Point point2= m_data.m_model->X_to_x(p2,extrinsic);
      dx /=Nx;
      Point3f X=p1;
      cv::Mat m_X(4,1,CV_32FC1),m_dX(4,1,CV_32FC1);
      m_X.at<float>(0,0)=X.x,m_X.at<float>(1,0)=X.y,m_X.at<float>(2,0)=X.z,m_X.at<float>(3,0)=1;
      m_dX.at<float>(0,0)=dx.x,m_dX.at<float>(1,0)=dx.y,m_dX.at<float>(2,0)=dx.z,m_dX.at<float>(3,0)=0;
      

      
      
      for(int lineIndex=0;lineIndex<=Nx;++lineIndex,m_X+=m_dX){
	cv::Mat W_X(4,1,CV_32FC1),C_X(4,1,CV_32FC1),P_x(3,1,CV_32FC1);
	cv::Mat _j_X_Pose = cv::Mat::zeros(2,6,CV_32FC1);
	cv::Mat _j_Energy_X=cv::Mat::zeros(1,2,CV_32FC1);
	C_X=extrinsic*m_X;
	P_x=intrinsic*C_X;
		
	/*get nearestEdgeDistance ponit*/
	Point point(P_x.at<float>(0,0)/P_x.at<float>(2,0),P_x.at<float>(1,0)/P_x.at<float>(2,0));
	Point nearstPoint = getNearstPointLocation(point);
		
	float dist2Edge=getDistanceToEdege(point1,point2,nearstPoint);

	if(distFrame.at<float>(nearstPoint)>=Config::configInstance().OPTIMIZER_NEASTP_THREHOLD||dist2Edge>Config::configInstance().MAX_VALIAD_DISTANCE||
	  distFrame.at<float>(point)>=Config::configInstance().OPTIMIZER_POINT_THREHOLD){
	  continue;
	}
	if(Config::configInstance().CV_LINE_P2NP){      
	  m_data.m_model->DisplayLine(point,nearstPoint,drawFrame,sqrt(dist2Edge));
	}
	//computeEnergy
	{	  
	  float _x=C_X.at<float>(0,0);
	  float _y=C_X.at<float>(0,1);
	  float _z=C_X.at<float>(0,2);
	  
	  //try compute another way 
	  {	  
	  J_X_Pose.at<float>(2*lineIndex,0)=m_calibration.fx()/_z; 
	  J_X_Pose.at<float>(2*lineIndex,1)=0;
	  J_X_Pose.at<float>(2*lineIndex,2)=-m_calibration.fx()*_x/(_z*_z);
	  J_X_Pose.at<float>(2*lineIndex,3)=-m_calibration.fx()*_x*_y/(_z*_z);
	  J_X_Pose.at<float>(2*lineIndex,4)=m_calibration.fx()*(1+_x*_x/(_z*_z));
	  J_X_Pose.at<float>(2*lineIndex,5)=m_calibration.fx()*_y/_z;

	  J_X_Pose.at<float>(2*lineIndex+1,0)=0;  
	  J_X_Pose.at<float>(2*lineIndex+1,1)=m_calibration.fy()/_z;      
	  J_X_Pose.at<float>(2*lineIndex+1,2)=-m_calibration.fy()*_y/(_z*_z);    
	  J_X_Pose.at<float>(2*lineIndex+1,3)=-m_calibration.fy()*(1+_y*_y*(1/(_z*_z)));
	  J_X_Pose.at<float>(2*lineIndex+1,4)=-m_calibration.fy()*_x*_y/(_z*_z);
	  J_X_Pose.at<float>(2*lineIndex+1,5)=m_calibration.fy()*_x/_z; 
	  
	  J_Energy_X.at<float>(0,2*lineIndex+0)=2*(point.x - nearstPoint.x);
	  J_Energy_X.at<float>(0,2*lineIndex+1)=2*(point.y - nearstPoint.y);

	  }
	  
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
		  

	  _j_Energy_X.at<float>(0,0)=2*(point.x - nearstPoint.x);
	  _j_Energy_X.at<float>(0,1)=2*(point.y - nearstPoint.y);
	  _j_X_Pose*=Config::configInstance().J_SIZE;
	  _j_Energy_X*=Config::configInstance().J_SIZE;
	  Mat _J=_j_Energy_X*_j_X_Pose;
	  Mat _J_T;
	  cv::transpose(_J,_J_T);
	  b+=_J_T*( (point.x - nearstPoint.x)*(point.x - nearstPoint.x)+(point.y - nearstPoint.y)*(point.y - nearstPoint.y));
	}
	


	j_X_Pose+=_j_X_Pose;
	j_Energy_X+=_j_Energy_X;  
      }

    }
    


  } 
  if(Config::configInstance().CV_LINE_P2NP){
    LOG(WARNING)<<"to draw drawFrame";
      m_data.m_model->DisplayCV(prePose,drawFrame);
      if(m_frameId%10==0){
	imshow("DRAW_LINE_P2NP",drawFrame);
      //   imshow("distMap",frame/255.f);
	waitKey(0);
      }
  }
  Mat J=J_Energy_X*J_X_Pose*Config::configInstance().J_SIZE;
  Mat J_T;  
  cv::transpose(J,J_T);
  A= J_T*J+lastA*lamda;//6*6
  A*=Config::configInstance().SIZE_A;


}

void Traker::constructEnergyFunction(const cv::Mat distFrame,const float* prePose,const cv::Mat &lastA,const int &lamda, cv::Mat &A, cv::Mat &b){
 cv::Mat j_X_Pose= cv::Mat::zeros(2,6,CV_32FC1);
  cv::Mat j_Energy_X=cv::Mat::zeros(1,2,CV_32FC1);
  
  Mat intrinsic = m_data.m_model->getIntrinsic();
  Mat extrinsic = Transformation::getTransformationMatrix(prePose);
  GLMmodel* model =m_data.m_model->GetObjModel();
  cv::Mat pos = m_data.m_model->getPos();
  int size=0;
  cv::Mat drawFrame;
  if(Config::configInstance().CV_LINE_P2NP){    
    cv::cvtColor(distFrame/255.f, drawFrame, CV_GRAY2BGR);
  }
  for(int i=0;i<model->numLines;++i){
    if(model->lines[i].tovisit){
      int v0=model->lines[i].vindices[0],v1=model->lines[i].vindices[1];
      Point3f p1= Point3f(pos.at<float>(0,v0-1),pos.at<float>(1,v0-1),pos.at<float>(2,v0-1));
      Point3f p2= Point3f(pos.at<float>(0,v1-1),pos.at<float>(1,v1-1),pos.at<float>(2,v1-1));
      Point3f dx=(p2-p1);
      int Nx=sqrt(dx.x*dx.x+dx.y*dx.y+dx.z*dx.z)/Config::configInstance().NX_LENGTH;
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

	if(distFrame.at<float>(nearstPoint)>=Config::configInstance().OPTIMIZER_NEASTP_THREHOLD||dist2Edge>Config::configInstance().MAX_VALIAD_DISTANCE||
	  distFrame.at<float>(point)>=Config::configInstance().OPTIMIZER_POINT_THREHOLD){
	  continue;
	}
	if(Config::configInstance().CV_LINE_P2NP){      
	  m_data.m_model->DisplayLine(point,nearstPoint,drawFrame,sqrt(dist2Edge));
	}
	//computeEnergy
	{	  
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
		  

	  _j_Energy_X.at<float>(0,0)=2*(point.x - nearstPoint.x);
	  _j_Energy_X.at<float>(0,1)=2*(point.y - nearstPoint.y);
	  _j_X_Pose*=Config::configInstance().J_SIZE;
	  _j_Energy_X*=Config::configInstance().J_SIZE;
  // 	_j_X_Pose*=(1.f/Nx);
  // 	_j_Energy_X*=(1.f/Nx);
	}
	
	Mat _J=_j_Energy_X*_j_X_Pose;
	Mat _J_T;
	cv::transpose(_J,_J_T);
	b+=_J_T*( (point.x - nearstPoint.x)*(point.x - nearstPoint.x)+(point.y - nearstPoint.y)*(point.y - nearstPoint.y));
	j_X_Pose+=_j_X_Pose;
	j_Energy_X+=_j_Energy_X;  
      }

    }

  } 
  /*
  LOG(WARNING)<<"_j_X_Pose\n"<<j_X_Pose;
  LOG(WARNING)<<"_j_Energy_X\n"<<j_Energy_X;*/
  
if(Config::configInstance().CV_LINE_P2NP){
  LOG(WARNING)<<"to draw drawFrame";
    m_data.m_model->DisplayCV(prePose,drawFrame);
    if(m_frameId%10==0){
      imshow("DRAW_LINE_P2NP",drawFrame);
    //   imshow("distMap",frame/255.f);
      waitKey(0);
    }
}
  
  Mat J(6,6,CV_32FC1),J_T(6,6,CV_32FC1);
  J=j_Energy_X*j_X_Pose *(1.0f/size)*(1.0f/size);  
//   printMat("J",J);
  cv::transpose(J,J_T);
//   float norm = A.diag();
  A= J_T*J+lastA*lamda;//6*6
  A*=Config::configInstance().SIZE_A;
//   printMat("A",A);

  
}

float Traker::getDistanceToEdege(const Point& e0, const Point& e1, const Point& v)
{
  return pow((e0.y-e1.y)*v.x +(e1.x-e0.x)*v.y+(e0.x*e1.y-e1.x*e0.y),2)/
	  (pow((e1.x-e0.x),2)+pow((e1.y-e0.y),2));
}
/*
int Traker::edfTracker(const float* prePose, const Mat& distMap,const  int NLrefine, float* newPose)
{
  Mat _distMap=distMap.clone();
  if (_distMap.isContinuous())
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
      int Nx = sqrt(dX.x*dX.x+dX.y*dX.y+dX.z*dX.z)/Config::configInstance().NX_LENGTH;
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
*/
void Traker::getCoarsePoseByPNP(const float *prePose, const Mat &distMap,float *coarsePose)
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
      int Nx = sqrt(dX.x*dX.x+dX.y*dX.y+dX.z*dX.z)/Config::configInstance().NX_LENGTH;

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
	float dist2Edge=getDistanceToEdege(point1,point2,nearst);
	if(distMap.at<float>(nearst)>=Config::configInstance().OPTIMIZER_NEASTP_THREHOLD/*||dist2Edge>Config::configInstance().MAX_VALIAD_DISTANCE*/){
	  continue;
	}
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

float Traker::computeEnergy(const cv::Mat& distFrame,const float * pose)
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
      int Nx = sqrt(dX.x*dX.x+dX.y*dX.y+dX.z*dX.z)/Config::configInstance().NX_LENGTH;

//       float Nx=100.f*norm;
      size+=Nx;
      dX /=Nx;
      Point3f X=p1;

      Point point1= m_data.m_model->X_to_x(p1,extrinsic);
      Point point2= m_data.m_model->X_to_x(p2,extrinsic);

       
      LOG(INFO)<< "v0:("<<v0<<") "<< point1.x<<" "<<point1.y<< " , v1:("<<v1<<") "<< point2.x<<" "<<point2.y<<std::endl;
      if(point1.x<0||point1.y<0||point2.x<0||point2.y<0||point1.x>=distFrame.size().width||point2.x>=distFrame.size().width||point1.y>=distFrame.size().height||point2.y>=distFrame.size().height){
	return INF;
      }
      float meanE_LINE=0;

       for(int i=0;i<=Nx;++i,X+=dX){
	 Point point= m_data.m_model->X_to_x(X,extrinsic);	 	 
	 Point nearst=getNearstPointLocation(point);
	 
	 float de2 = /*getDistance2ToEdege(point1,point2,nearst)+*/distFrame.at<float>(point);
	 float dist2Edge=getDistanceToEdege(point1,point2,nearst);
	 //enlarge influence of 255
	 if(de2==255.f){
	    de2*=10;
	 }
// 	 if(dist2Edge>Config::configInstance().MAX_VALIAD_DISTANCE){
// 	   size--;
// 	   continue;
// 	 }
// 	 de2*=(1-pow( dist2Edge/Config::configInstance().MAX_VALIAD_DISTANCE,2));
	 LOG(INFO)<< i<<"th point :"<<point.x<<" "<<point.y<< "  energy: " << distFrame.at<float>(point)<<" Distance energy: "<<de2<<"  np: "<<nearst.x<<" "<<nearst.y<<" dist2Edge ="<< dist2Edge;	 
//	 if(DX<Config::configInstance().THREHOLD_DX||DX<meanE_LINE*meanE_LINE){
	  energy+=de2;
//	 }
  
       }

    }
  } 

  N_Points=size;
  energy*=1.0f/size;
  LOG(INFO)<<"Total mean Energy = "<<energy;

  return energy;
  
}

float Traker::nearestEdgeDistance(const cv::Point & point,const std::vector<m_img_point_data>  &edge_points,cv::Point &nPoint ,const bool printP)
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
  return Config::configInstance().ENERGY_SIZE*nearstD; 
}
void Traker::updateState(const cv::Mat&distFrame, const Mat& dX, const Transformation& old_Transformation, Transformation& new_transformation,float &e2_new )
{
	  if(Config::configInstance().USE_MY_TRANSFORMATION){
	    new_transformation.setPose(old_Transformation.Pose());
	    new_transformation.xTransformation(dX);
	    e2_new = computeEnergy(distFrame, new_transformation.Pose());
	  }
	  else if(Config::configInstance().USE_SOPHUS)
	  {
	    Sophus::SE3 T_SE3;
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
	    new_transformation.setPoseFromTransformationMatrix(new_T_SE3.matrix());
	    e2_new = computeEnergy(distFrame, new_transformation.Pose());
	  }/*else{
	    float newPose[6];
	    UpdateStateLM(dX,m_Transformation.Pose(),newPose);	
	    new_transformation.setPose(newPose);
	    e2_new = computeEnergy(distFrame, new_transformation.Pose());
	  }*/
}

void Traker::UpdateStateLM(const cv::Mat &dx, const float * pose_Old, Transformation &transformation_New)
{
  
  transformation_New.setPose(pose_Old);  
   LOG(INFO)<<"pose_Old"<<transformation_New.Pose();
  transformation_New.xTransformation(dx);
   LOG(INFO)<<"transformation_New"<<transformation_New.transformationMatrix();

}

//update C1+dx ->  C2
void Traker::UpdateStateLM(const cv::Mat &dx, const float * pose_Old, float * pose_New)
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
void Traker::getDistMap(const Mat& frame)
{
    auto edgeDetector = std::make_unique<ED::ImgProcession>();
    edgeDetector->getDistanceTransform(frame,Config::configInstance().IMG_DIST_MASK_SIZE,distFrame,locationsMat);
}


}