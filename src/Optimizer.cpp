#include <fstream>      //C++
#include <iostream>
#include<glog/logging.h>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include "ObjectDetector/Optimizer.h"
using namespace OD;
const int MAX_ITERATIN_NUM =50;
const float THREHOLD_ENERGY = 160.0f;
const float DX_SIZE = 1.f;
const float NX_LENGTH = 0.01f;
const float ENERGY_SIZE = 1.0f;
const float J_SIZE = 0.00001f;
const float LM_STEP =10 ;
const float INIT_LAMDA =1;
const float SIZE_A =1;
const float INF =1e10;
const float THREHOLD_DX= 1e7;


#define FACTOR_DEG_TO_RAD 0.01745329252222222222222222222222f
#define PRIOR_MAX_DEVIATION_CAMERA_HEIGHT			0.5f
#define PRIOR_MAX_DEVIATION_CAMERA_ROTATION_XY			5.0f
#define PRIOR_MAX_DEVIATION_CAMERA_ROTATION_Z			30.0f
#define SIMULATION_MAX_ROTATIONS				3
#define SIMULATION_MAX_ROTATION_RATIO			0.3f
#define SIMULATION_MAX_TRANSLATIONS				1
#define SIMULATION_MAX_TRANSLATION_RATIO		0.5f
static void printMat(std::string name, cv::Mat M){
//   LOG(INFO)<<name<<std::endl;;
  cout<<name<<endl;
  cout <<M<<endl;
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


cv::Mat Optimizer::optimizingLM(const Mat& prePose, const Mat& frame, const Mat& locations, const int frameId)
{
   int64 time0 = cv::getTickCount();
  {
    LOG(WARNING)<<" ";
    LOG(WARNING)<<"frameId = "<<frameId;
    _locations=(int *)locations.data;
    _col=frame.cols;
    _row=frame.rows;
//     printMat("frame",frame);
//     imshow("frame",frame);
//     waitKey(0);
    cv::Mat A_I=cv::Mat::eye(6,6,CV_32FC1);
    float lamda=INIT_LAMDA;
    std::vector<cv::Point> nPoints;
    m_Transformation.setPose(prePose);
//     cv::Mat _prePose =prePose.clone();
    int itration_num=0;
    cv::Mat newPose=cv::Mat::zeros(1,6,CV_32FC1);  
    Transformation newTransformation;
    m_data.m_model->setVisibleLinesAtPose(m_Transformation.Pose());

    
    float e2 = computeEnergy(frame, m_Transformation.Pose());

    while(++itration_num<MAX_ITERATIN_NUM){      
      	  
      LOG(INFO)<<"a itration_num = " <<itration_num;

      
      cv::Mat _A= cv::Mat::zeros(6,6,CV_32FC1),b= cv::Mat::zeros(6,1,CV_32FC1),A= cv::Mat::zeros(6,6,CV_32FC1);
      if(e2<THREHOLD_ENERGY){
	LOG(INFO)<<"good init ,no need to optimize! energy = "<<e2;
	return m_Transformation.Pose();
      }else{      	
	LOG(WARNING)<<"to optimize with energy = "<<e2;	
      }

      constructEnergyFunction(frame,m_Transformation.Pose(),A_I,lamda, _A,b);
      _A/=abs(_A.at<float>(0,0));

      A=_A+A_I*lamda;
      LOG(INFO)<<"A_I\n"<<A_I;
      LOG(INFO)<<"lamda = "<<lamda;
      LOG(INFO)<<"A\n"<<A;

      Mat A_inverse ;
      cv::invert(A,A_inverse);
    //  printMat("b",b);
      b/=abs(b.at<float>(0,0));
      LOG(INFO)<<"A_inverse\n"<<A_inverse;      
      LOG(INFO)<<"b\n"<<b;
      Mat dX = A_inverse*b*DX_SIZE;
      LOG(INFO)<<"dX "<<dX;
      
      UpdateStateLM(dX,m_Transformation.Pose(),newPose);
      float e2_new = computeEnergy(frame, newPose);
      LOG(INFO)<<"_prePose "<<m_Transformation.Pose();

      while(e2_new>e2){	  
	  LOG(INFO)<<"sorry!!!Not to optimize! e2 :"<<e2<<" e2_new: "<<e2_new<<" \n";
	  lamda*=LM_STEP;
	  A=_A+A_I*lamda;
	  cv::invert(A,A_inverse);
	  
	  b/=abs(b.at<float>(0,0));
// 	  LOG(INFO)<<"A_inverse\n"<<A_inverse;      
// 	  LOG(INFO)<<"b\n"<<b;
	  Mat dX = A_inverse*b*DX_SIZE;
// 	  LOG(INFO)<<"dX "<<dX;
  	  UpdateStateLM(dX,m_Transformation.Pose(),newPose);	   
 	  UpdateStateLM(dX,m_Transformation.Pose(),newTransformation);
// 	  LOG(INFO)<<"newPose_T\n"<<Transformation::getTransformationMatrix(newPose);
	  float lastE2=e2_new;

	  
 	  e2_new = computeEnergy(frame, newPose);
  	  LOG(WARNING)<<"newPose"<<newPose<<"  e2_new(newPose) = "<<e2_new;
	  
 	  e2_new = computeEnergy(frame, newTransformation.Pose());
 	  LOG(WARNING)<<"newTransformation.Pose():"<<newTransformation.Pose()<<"e2_new(newTransformation) = "<<e2_new;
	  itration_num++;
	  if(itration_num>MAX_ITERATIN_NUM||abs(lastE2-e2_new)<1e-3){
	    LOG(INFO)<<"to much itration_num!";
	    return m_Transformation.Pose();    
	  }
	
      }
      	     
      if(e2_new>=e2)
	break;
      
      LOG(WARNING)<<"good !To optimize! e2 :"<<e2<<" e2_new: "<<e2_new;
      LOG(INFO)<<"dX "<<dX;
      LOG(INFO)<<"newPose "<<newPose;
      A_I=A.clone();
      A_I/=abs(A_I.at<float>(0,0));
      computeEnergy(frame,newPose);
//       m_Transformation.setPose(newPose);         
      m_Transformation.setPose(newTransformation.Pose());

      e2= e2_new;
      lamda=INIT_LAMDA;
      
      if(e2_new<THREHOLD_ENERGY){
	LOG(WARNING)<<"succees optimize!";
	return m_Transformation.Pose();
	
      }
      
    }
    LOG(INFO)<<"to much itration_num!";
    return m_Transformation.Pose();
    
  }
  
  int64 teim0 = cv::getTickCount();
}



cv::Point Optimizer::getNearstPointLocation(const cv::Point &point){
  int x= point.y;
  int y= point.x;   			
  while(_locations[y + _col * x]!=x||_locations[_row*_col+y + _col * x]!=y){
    x=_locations[y + _col * x];
    y=_locations[_row*_col+y + _col* x];
//      cout<<"move to: "<<x<<" y: "<<y<<endl;
   }
//    cout<<"End to: x "<<x<<" y: "<<y<<endl;
   return Point(y,x);
}
void Optimizer::constructEnergyFunction(const cv::Mat frame,const cv::Mat prePose,const cv::Mat &lastA,const int &lamda, cv::Mat &A, cv::Mat &b){
  cv::Mat j_X_Pose= cv::Mat::zeros(2,6,CV_32FC1);
  cv::Mat j_Energy_X=cv::Mat::zeros(1,2,CV_32FC1);
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
	Point point(P_x.at<float>(0,0)/P_x.at<float>(2,0),P_x.at<float>(1,0)/P_x.at<float>(2,0));
	Point nearstPoint = getNearstPointLocation(point);
//	printf("point: %d %d \n",point.x,point.y);
	_j_Energy_X.at<float>(0,0)=2*(point.x- nearstPoint.x*point.x);
;
	_j_Energy_X.at<float>(0,1)=2*(point.y- nearstPoint.y*point.y);
;
// 	_j_Energy_X.at<float>(0,0)=2*(point.x - nearstPoint.x);
// 	_j_Energy_X.at<float>(0,1)=2*(point.y - nearstPoint.y);
	_j_X_Pose*=J_SIZE;
	_j_Energy_X*=J_SIZE;
	
// 	_j_X_Pose*=(1.f/Nx);
// 	_j_Energy_X*=(1.f/Nx);
	Mat _J=_j_Energy_X*_j_X_Pose;
	Mat _J_T;
	cv::transpose(_J,_J_T);
// 	b+=_J_T*(sqrt(getDistanceToEdege(point1,point2,point)));
	b+= _J_T*frame.at<float>(point);
// 	b+=_J_T*( (point.x - nearstPoint.x)*(point.x - nearstPoint.x)+(point.y - nearstPoint.y)*(point.y - nearstPoint.y));
//     printf("%d : point  %d %d , npoint %d %d\n",i,m_data.m_pointset.m_img_points[i-1].x,m_data.m_pointset.m_img_points[i-1].y,nPoints[i-1].x,nPoints[i-1].y);
	

	j_X_Pose+=_j_X_Pose;
	j_Energy_X+=_j_Energy_X;  
      }

    }

  } 
  LOG(INFO)<<"_j_X_Pose\n"<<j_X_Pose;
  LOG(INFO)<<"_j_Energy_X\n"<<j_Energy_X;
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


float Optimizer::computeEnergy(const cv::Mat& frame,const cv::Mat& pose)
{
 m_data.m_model->GetImagePoints(pose, m_data.m_pointset);
  float energy =0;
  int size =0;
  Mat intrinsic = m_data.m_model->getIntrinsic();
  Mat extrinsic = m_data.m_model->GetPoseMatrix(pose);
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
	 meanE_LINE+=de2;
//  	 LOG(INFO)<< i<<"th point :"<<point.x<<" "<<point.y<< "  energy: " <<" Distance energy2: "<<de2<<"  np: "<<nearst.x<<" "<<nearst.y<<" nEnergy = "<<frame.at<float>(nearst);	 
       }
       model->lines[i].energy=meanE_LINE;
       meanE_LINE/=Nx;
       LOG(INFO)<<"meanE_LINE = "<<meanE_LINE;
       X=p1;
       for(int i=0;i<=Nx;++i,X+=dX){
	 Point point= m_data.m_model->X_to_x(X,extrinsic);	 	 
	 Point nearst=getNearstPointLocation(point);
	 
	 float de2 = /*getDistance2ToEdege(point1,point2,nearst)+*/frame.at<float>(point);
	 float DX=pow(de2-meanE_LINE,2); 	
	 LOG(INFO)<< i<<"th point :"<<point.x<<" "<<point.y<< "  energy: " << frame.at<float>(point)<<" Distance energy: "<<de2<<"  np: "<<nearst.x<<" "<<nearst.y<<" DX = "<<DX;	 
	 if(DX<THREHOLD_DX||DX<meanE_LINE*meanE_LINE){
	  meanDX+=DX;
	  energy+=de2;
	 }
  
       }

    }
  } 

	
  meanDX*=1.0f/size;
  energy*=1.0f/size;
  LOG(INFO)<<"meanDX = "<<meanDX;
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
void Optimizer::UpdateStateLM(const cv::Mat &dx, const cv::Mat &pose_Old, Transformation &transformation_New)
{
  
  transformation_New.setPose(pose_Old);  
   LOG(INFO)<<"pose_Old"<<transformation_New.Pose();
  transformation_New.xTransformation(dx);
   LOG(INFO)<<"transformation_New"<<transformation_New.transformationMatrix();

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
//   LOG(INFO)<<"pose_Old "<<pose_Old;
//   LOG(INFO)<<"pose_New "<<pose_New;
}


