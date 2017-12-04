#include <iostream>
#include "Traker/Model.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>
#include <opencv/cv.h>
#include <vector>
#include "Traker/Render.h"
#include "Traker/Transformation.h"
#include <omp.h>
#include<glog/logging.h>
using namespace OD;

extern ORD::Render g_render;
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
Model::Model(const Config& config) : m_bb_points(4,8,CV_32FC1),m_model( config.model),m_radius ( glmMaxRadius(m_model)),m_calibration ( config.camCalibration),	m_width ( config.VIDEO_WIDTH),
	m_height ( config.VIDEO_HEIGHT)
{
	getVisibleLines();
	
	intrinsic=cv::Mat(3,4,CV_32FC1);
	intrinsic.at<float>(0,0)=m_calibration.fx(); intrinsic.at<float>(0,1)=0; intrinsic.at<float>(0,2)=m_calibration.cx(); intrinsic.at<float>(0,3)=0;
	intrinsic.at<float>(1,0)=0; intrinsic.at<float>(1,1)=m_calibration.fy(); intrinsic.at<float>(1,2)=m_calibration.cy(); intrinsic.at<float>(1,3)=0;
	intrinsic.at<float>(2,0)=0; intrinsic.at<float>(2,1)=0; intrinsic.at<float>(2,2)=1; intrinsic.at<float>(2,3)=0;
	
	modelPos =cv::Mat (4,m_model->numvertices,CV_32FC1);
	for(int i=1; i<=m_model->numvertices; i++)
	{
	  modelPos.at<float>(0,i-1) = m_model->vertices[3*(i)+0];
	  modelPos.at<float>(1,i-1) = m_model->vertices[3*(i)+1];
	  modelPos.at<float>(2,i-1) = m_model->vertices[3*(i)+2];
	  modelPos.at<float>(3,i-1) = 1;
	}
	
	//camera intrinsic
	intrinsic=cv::Mat(3,4,CV_32FC1);
	intrinsic.at<float>(0,0)=m_calibration.fx(); intrinsic.at<float>(0,1)=0; intrinsic.at<float>(0,2)=m_calibration.cx(); intrinsic.at<float>(0,3)=0;
	intrinsic.at<float>(1,0)=0; intrinsic.at<float>(1,1)=m_calibration.fy(); intrinsic.at<float>(1,2)=m_calibration.cy(); intrinsic.at<float>(1,3)=0;
	intrinsic.at<float>(2,0)=0; intrinsic.at<float>(2,1)=0; intrinsic.at<float>(2,2)=1; intrinsic.at<float>(2,3)=0;
}

Model::~Model()
{
	if(m_model)
		glmDelete(m_model);
}

void Model::LoadGLMModel(const std::string& filename)
{
	m_model = glmReadOBJ(const_cast<char*>(filename.c_str()));
	m_radius = glmMaxRadius(m_model);	
	if(!m_model)
		return;
}

GLMmodel* Model::GetObjModel()
{
	return m_model;
}


void Model::GetImagePoints(const float* prepose, PointSet& pointset)
{
	pointset.m_img_points.clear();
	pointset.m_img_points_f.clear();	
	cv::Mat extrinsic = Transformation::getTransformationMatrix(prepose);
	
	cv::Mat result(3,m_model->numvertices,CV_32FC1);

	result = intrinsic*extrinsic*modelPos;
	/*
	printf("samplePoints.size() =%d \n",samplePoints.size());*/
	//normalize
	for(int i=0; i<m_model->numvertices; i++)
	{
		float u = result.at<float>(0,i)/result.at<float>(2,i);
		float v = result.at<float>(1,i)/result.at<float>(2,i);
		if(u>=0 && u<m_width && v>=0 && v<m_height)
		{
// 			printf("u = %f, v= %f\n",u,v);
			pointset.m_img_points.push_back(cv::Point(u,v));
			pointset.m_img_points_f.push_back(cv::Point2f(u,v));
		}
	}
}


// PointSet& Model::GetVisibleModelPointsCV(const cv::Mat& prepose, int pointnum)
// {
// 	m_point_set.Clear();
// 	FilterModel(prepose,pointnum);
// 	return m_point_set;
// }


static inline GLvoid
glmCross(GLfloat* u, GLfloat* v, GLfloat* n)
{
    assert(u); assert(v); assert(n);
    
    n[0] = u[1]*v[2] - u[2]*v[1];
    n[1] = u[2]*v[0] - u[0]*v[2];
    n[2] = u[0]*v[1] - u[1]*v[0];
}

/* glmNormalize: normalize a vector
 *
 * v - array of 3 GLfloats (GLfloat v[3]) to be normalized
 */
static inline GLvoid
glmNormalize(GLfloat* v)
{
    GLfloat l;
    
    assert(v);
    
    l = (GLfloat)sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    v[0] /= l;
    v[1] /= l;
    v[2] /= l;
}
void Model::getVisibleLines()
{
   cv::Mat _pos(3,m_model->numvertices+1,CV_32FC1);


   for(int i=0; i<=m_model->numvertices; i++)
   {		
     _pos.at<float>(0,i) = m_model->vertices[3*(i)+0];	
     _pos.at<float>(1,i) = m_model->vertices[3*(i)+1];	
     _pos.at<float>(2,i) = m_model->vertices[3*(i)+2];
     
  }
  float u[3],n[3],v[3];
  for (int i = 0; i < m_model->numtriangles; i++) {
    u[0] = _pos.at<float>(0,m_model->triangles[i].vindices[1]) - _pos.at<float>(0,m_model->triangles[i].vindices[0]);  
    u[1] = _pos.at<float>(1,m_model->triangles[i].vindices[1]) - _pos.at<float>(1,m_model->triangles[i].vindices[0]);
    u[2] = _pos.at<float>(2,m_model->triangles[i].vindices[1]) - _pos.at<float>(2,m_model->triangles[i].vindices[0]);
    v[0] = _pos.at<float>(0,m_model->triangles[i].vindices[2]) - _pos.at<float>(0,m_model->triangles[i].vindices[0]);
    v[1] = _pos.at<float>(1,m_model->triangles[i].vindices[2]) - _pos.at<float>(1,m_model->triangles[i].vindices[0]);
    v[2] = _pos.at<float>(2,m_model->triangles[i].vindices[2]) - _pos.at<float>(2,m_model->triangles[i].vindices[0]);
    glmCross(u, v, n);
    glmNormalize(n); 
    /* get visible line*/
    {			
      if(m_model->lines[m_model->triangles[i].lindices[0]].e1 == 0){				
	m_model->lines[m_model->triangles[i].lindices[0]].e1 = 1;
				for(int j=0;j<3;j++){
				  m_model->lines[m_model->triangles[i].lindices[0]].n1[j] =n[j] ;
				}
			}
			else{
				m_model->lines[m_model->triangles[i].lindices[0]].e2 = 1;
				for(int j=0;j<3;j++){
				  m_model->lines[m_model->triangles[i].lindices[0]].n2[j] =n[j] ;
				}
			}

			if(m_model->lines[m_model->triangles[i].lindices[1]].e1 == 0){
				m_model->lines[m_model->triangles[i].lindices[1]].e1 = 1;
				for(int j=0;j<3;j++){
				  m_model->lines[m_model->triangles[i].lindices[1]].n1[j] =n[j] ;
				}
			}
			else{
				m_model->lines[m_model->triangles[i].lindices[1]].e2 = 1;
				for(int j=0;j<3;j++){
				  m_model->lines[m_model->triangles[i].lindices[1]].n2[j] =n[j] ;
				}
			}

			if(m_model->lines[m_model->triangles[i].lindices[2]].e1 == 0){
				m_model->lines[m_model->triangles[i].lindices[2]].e1 = 1;
				for(int j=0;j<3;j++){
				  m_model->lines[m_model->triangles[i].lindices[2]].n1[j] =n[j] ;
				}
			}
			else{
				m_model->lines[m_model->triangles[i].lindices[2]].e2 = 1;
				for(int j=0;j<3;j++){
				  m_model->lines[m_model->triangles[i].lindices[2]].n2[j] =n[j] ;
				}
			}
/*      if(m_model->lines[m_model->triangles[i].lindices[0]].e1 == 0){
	m_model->lines[m_model->triangles[i].lindices[0]].e1 = 1;
	memcpy( m_model->lines[m_model->triangles[i].lindices[0]].n1,n,sizeof(n));	
      }
      else{
	m_model->lines[m_model->triangles[i].lindices[0]].e2 = 1;
	memcpy( m_model->lines[m_model->triangles[i].lindices[0]].n2,n,sizeof(n));	
      }
      if(m_model->lines[m_model->triangles[i].lindices[1]].e1 == 0){
	m_model->lines[m_model->triangles[i].lindices[1]].e1 = 1;
	memcpy( m_model->lines[m_model->triangles[i].lindices[1]].n1,n,sizeof(n));	

      }
      else{
	m_model->lines[m_model->triangles[i].lindices[1]].e2 = 1;
		memcpy( m_model->lines[m_model->triangles[i].lindices[1]].n1,n,sizeof(n));	
	
      }
      if(m_model->lines[m_model->triangles[i].lindices[2]].e1 == 0){
	m_model->lines[m_model->triangles[i].lindices[2]].e1 = 1;
	memcpy( m_model->lines[m_model->triangles[i].lindices[2]].n1,n,sizeof(n));	

      }
      else{
	m_model->lines[m_model->triangles[i].lindices[2]].e2 = 1;
	memcpy( m_model->lines[m_model->triangles[i].lindices[2]].n1,n,sizeof(n));	
	
      } */     
    }
  }
  for(int i=0; i<m_model->numLines; ++i)
  {
//     m_model->lines[i].visible=true;
     m_model->lines[i].visible=!isSameNormal(m_model->lines[i].n1, m_model->lines[i].n2);
  }
  
}
int Model::crossProductNorm(const Point &p, const Point &p1)
{
//   int x1= p.x,x2=p1.x,y1=p.y,y2=p1.y;
//   printf("x1 %d  y1 %d  x2 %d y2 %d ",x1,y1,x2,y2);
//   int t= ;
//   printf(" t =%d\n",t);
  return p.x*p1.y - p.y*p1.x;
}

const cv::Mat& Model::getIntrinsic() const
{
  return intrinsic;
}

void Model::getVisualableVertices(const float * pose, cv::Mat& vis_vertices) {
  using namespace cv;

	int visualable_line_count = 0;
	std::vector<cv::Point3f> Xs; 
	for (size_t i = 0; i<m_model->numLines; i++) {
		if ((m_model->lines[i].tovisit )) {
			GLuint v0 = m_model->lines[i].vindices[0];
			GLuint v1 = m_model->lines[i].vindices[1];
			Point3f p1= Point3f(m_model->vertices[3 * v0],m_model->vertices[3 * v0 + 1],m_model->vertices[3 * v0 + 2]);
			Point3f p2= Point3f(m_model->vertices[3 * v1],m_model->vertices[3 * v1 + 1],m_model->vertices[3 * v1 + 2]);
			Point3f dX=(p2-p1);      
			int Nx = sqrt(dX.x*dX.x+dX.y*dX.y+dX.z*dX.z)/Config::configInstance().NX_LENGTH;
			LOG(INFO)<<"visible lines "<<m_model->lines[i].vindices[0]<<" "<<m_model->lines[i].vindices[1]<<" Nx = "<<Nx;

			if(Nx<1){
			  Nx=1;
			}
			dX/=Nx;
			for(int j=0;j<=Nx;j++){
			  Xs.push_back(p1+dX*j);
			}			
		}
	}
	visualable_line_count=Xs.size();
	LOG(INFO)<<"visualable_line_count = "<<visualable_line_count;
	cv::Mat pos(4, visualable_line_count, CV_32FC1);
	for(int i=0;i<visualable_line_count;i++){
	  pos.at<float>(0, i) =	Xs[i].x;
	  pos.at<float>(1, i) = Xs[i].y;
	  pos.at<float>(2, i) = Xs[i].z;
	  pos.at<float>(3, i) = 1;
	}
	vis_vertices = pos;
}

void Model::setVisibleLinesAtPose(const float * pose)
{
   
  using namespace cv;
  for(int i=0;i<m_model->numLines;i++)
    m_model->lines[i].tovisit=false; 
  cv::Mat pt_in_cam(3, m_model->numvertices+1, CV_32FC1);

  cv::Mat extinsic(3, 4, CV_32FC1);
  extinsic= Transformation::getTransformationMatrix(pose);
  pt_in_cam = extinsic * modelPos;
  
  float u[3], v[3], n[3], center[3];
  for (size_t i = 0; i < m_model->numtriangles; i++) {
    //compute the norm of the triangles
    u[0] = pt_in_cam.at<float>(0, m_model->triangles[i].vindices[1]-1) - pt_in_cam.at<float>(0, m_model->triangles[i].vindices[0]-1);
    u[1] = pt_in_cam.at<float>(1, m_model->triangles[i].vindices[1]-1) - pt_in_cam.at<float>(1, m_model->triangles[i].vindices[0]-1);
    u[2] = pt_in_cam.at<float>(2, m_model->triangles[i].vindices[1]-1) - pt_in_cam.at<float>(2, m_model->triangles[i].vindices[0]-1);
    
    v[0] = pt_in_cam.at<float>(0, m_model->triangles[i].vindices[2]-1) - pt_in_cam.at<float>(0, m_model->triangles[i].vindices[0]-1);
    v[1] = pt_in_cam.at<float>(1, m_model->triangles[i].vindices[2]-1) - pt_in_cam.at<float>(1, m_model->triangles[i].vindices[0]-1);
    v[2] = pt_in_cam.at<float>(2, m_model->triangles[i].vindices[2]-1) - pt_in_cam.at<float>(2, m_model->triangles[i].vindices[0]-1);

    glmCross(u, v, n);
    glmNormalize(n);
      
    //get center of triangle center[3]
    {
      center[0] = 1.0/3*(pt_in_cam.at<float>(0, m_model->triangles[i].vindices[0]-1) + pt_in_cam.at<float>(0, m_model->triangles[i].vindices[1]-1) + pt_in_cam.at<float>(0, m_model->triangles[i].vindices[2]-1));
      center[1] = 1.0/3*(pt_in_cam.at<float>(1, m_model->triangles[i].vindices[0]-1) + pt_in_cam.at<float>(1, m_model->triangles[i].vindices[1]-1) + pt_in_cam.at<float>(1, m_model->triangles[i].vindices[2]-1));
      center[2] = 1.0/3*(pt_in_cam.at<float>(2, m_model->triangles[i].vindices[0]-1) + pt_in_cam.at<float>(2, m_model->triangles[i].vindices[1]-1) + pt_in_cam.at<float>(2, m_model->triangles[i].vindices[2]-1));
    }
    glmNormalize(center);
    
    //judge the whether the line is visible or not
    float cross = n[0] * center[0] + n[1] * center[1] + n[2] * center[2];
    if (cross < 0.0f) {		  
      m_model->lines[m_model->triangles[i].lindices[0]].tovisit=(true&m_model->lines[m_model->triangles[i].lindices[0]].visible);
      m_model->lines[m_model->triangles[i].lindices[1]].tovisit=(true&m_model->lines[m_model->triangles[i].lindices[1]].visible);
      m_model->lines[m_model->triangles[i].lindices[2]].tovisit=(true&m_model->lines[m_model->triangles[i].lindices[2]].visible);
      
    }
    
  }

}
Point Model::X_to_x(const Point3f &X,const Mat &extrisic)
{
  Mat P(4,1,CV_32FC1);
  Mat res(3,1,CV_32FC1);
  P.at<float>(0,0)=X.x;
  P.at<float>(1,0)=X.y;
  P.at<float>(2,0)=X.z;
  P.at<float>(3,0)=1;
  res=intrinsic*extrisic*P;
  
  return Point(res.at<float>(0,0)/res.at<float>(2,0),res.at<float>(1,0)/res.at<float>(2,0));
  
}

void Model::DisplayLine(const cv::Point& p1,const cv::Point& p2, cv::Mat& frame,const float &radius)
{
  
  float drawRadius = radius<Config::configInstance().CV_CIRCLE_RADIUS?Config::configInstance().CV_CIRCLE_RADIUS:radius;
  if(radius<Config::configInstance().CV_CIRCLE_RADIUS){
    cv::circle(frame,p1,drawRadius,Scalar(0,0,255));  
  }else{
    cv::circle(frame,p1,drawRadius,Scalar(255,0,0));  
  } 
  cv::line(frame,p1,p2,cv::Scalar(0,255,0),1,CV_AA);
}
void Model::Project(const float* pose, const Mat& visible_Xs,  Mat &visible_xs)
{
  	
  cv::Mat extrinsic = Transformation::getTransformationMatrix(pose);
  visible_xs=intrinsic*extrinsic*visible_Xs;
  for (int i = 0; i < visible_xs.cols; ++i) {
    float dz = 1.0f/visible_xs.at<float>(2, i);
    visible_xs.at<float>(0, i) *= dz;
    visible_xs.at<float>(1, i) *= dz;
  }
}

void Model::DisplayCV(const float * pose,const cv::Scalar &color, cv::Mat& frame)
{
  
	//camera extrinsic
	cv::Mat extrinsic = Transformation::getTransformationMatrix(pose);

	//extract the points which can be visible or in the edge of object
	std::vector<cv::Point> vertexIndexs;
	std::vector<cv::Point3f> points_3d;
	setVisibleLinesAtPose(pose);
	for(int i=0; i<m_model->numLines; i++)
	{
// 		if((m_model->lines[i].e1 == 1 && m_model->lines[i].e2 == 0) || (m_model->lines[i].e1==0 && m_model->lines[i].e2==1) || (m_model->lines[i].e1==1 && m_model->lines[i].e2==1))
// 	  if(!isSameNormal(m_model->lines[i].n1, m_model->lines[i].n2)){
	  if(m_model->lines[i].tovisit){
		GLuint v0 = m_model->lines[i].vindices[0];
		GLuint v1 = m_model->lines[i].vindices[1];
		
		
		Point3f p1= Point3f(modelPos.at<float>(0,v0-1),modelPos.at<float>(1,v0-1),modelPos.at<float>(2,v0-1));
		Point3f p2= Point3f(modelPos.at<float>(0,v1-1),modelPos.at<float>(1,v1-1),modelPos.at<float>(2,v1-1));

		points_3d.push_back(p1);
		points_3d.push_back(p2);
			
		vertexIndexs.push_back(cv::Point(v0,v1));

		}
		m_model->lines[i].e1 = 0; m_model->lines[i].e2 = 0;
	}

	//compute the image coordinate of visible points
	cv::Mat pos(4,2*vertexIndexs.size(),CV_32FC1);
	cv::Mat result(3,2*vertexIndexs.size(),CV_32FC1);

	for(int i=0; i<(int)vertexIndexs.size(); i++)
	{
		pos.at<float>(0,2*i) = points_3d[2*i].x;
		pos.at<float>(1,2*i) = points_3d[2*i].y;
		pos.at<float>(2,2*i) = points_3d[2*i].z;
		pos.at<float>(3,2*i) = 1;
		pos.at<float>(0,2*i+1) = points_3d[2*i+1].x;
		pos.at<float>(1,2*i+1) = points_3d[2*i+1].y;
		pos.at<float>(2,2*i+1) = points_3d[2*i+1].z;
		pos.at<float>(3,2*i+1) = 1;
	}
	result = intrinsic*extrinsic*pos;
	LOG(INFO)<<"---to draw visit line";
	//display the visible lines
	for(int i=0; i<vertexIndexs.size(); i++)
	{
		int u1 = result.at<float>(0,2*i)/result.at<float>(2,2*i);
		int v1 = result.at<float>(1,2*i)/result.at<float>(2,2*i);
		
		int u2 = result.at<float>(0,2*i+1)/result.at<float>(2,2*i+1);
		int v2 = result.at<float>(1,2*i+1)/result.at<float>(2,2*i+1);
				 
// 		LOG(INFO)<<"frame.cols = "<<frame.cols<<" , frame.rows= "<<frame.rows; 

		if(u1 >=0 && u1<frame.cols && v1 >=0 && v1<=frame.rows && u2 >=0 && u2<frame.cols && v2>=0 && v2<=frame.rows)
		{
// 		  LOG(INFO)<< "v0:"<< u1<<" "<<v1<< " , v1:"<< u2<<" "<<v2<<std::endl;

		  cv::line(frame,cv::Point(u1,v1),cv::Point(u2,v2),color,1,CV_AA);
		}
	}
}


cv::Point3f Model::getPos_E(int e)
{
  float x0 = m_model->vertices[3*e];
  float y0 = m_model->vertices[3*e+1];
  float z0 = m_model->vertices[3*e+2];
  return cv::Point3f(x0,y0,z0);
}
bool Model::isSameNormal(const float* n1, const float* n2)
{
  int k=1;
  float n[3];
  glmCross((GLfloat *)n1,(GLfloat *)n2,n);
  return (abs(n[0])<1e-2&&abs(n[1])<1e-2&&abs(n[2])<1e-2);
  
}
const Mat& Model::getPos() const
{
  return modelPos;
}
void Model::getContourPointsAndIts3DPoints(const  float *pose,std::vector<cv::Point3d> &contour_Xs,std::vector<cv::Point2d> &contour_xs){
  LOG(INFO)<<"getContourPointsAndIts3DPoints";
  Mat visible_Xs,visible_xs;
  getVisualableVertices(pose,visible_Xs);
  Project(pose, visible_Xs, visible_xs);
  LOG(INFO)<<"visible_Xs.size "<<visible_Xs.size();
  cv::Mat img1=cv::Mat::zeros(m_height, m_width, CV_32SC1);
  for (int i = 0; i < visible_xs.cols; ++i) {
    cv::Point pt(visible_xs.at<float>(0, i), visible_xs.at<float>(1, i));
//     LOG(INFO)<<i + 1<<" pt : "<<pt;
    if (pointInFrame(pt)){
      img1.at<int>(pt) = i + 1;
    }
  }
      
  std::vector<std::vector<cv::Point> > contours; 

  cv::Mat line_img = cv::Mat::zeros(Config::configInstance().VIDEO_HEIGHT, Config::configInstance().VIDEO_WIDTH , CV_8UC1);
  DisplayCV(pose, cv::Scalar(255, 255, 255),line_img);  
  cv::findContours(line_img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

//   cv::Mat mask_img = cv::Mat::zeros(Config::configInstance().VIDEO_HEIGHT,Config::configInstance().VIDEO_WIDTH, CV_8UC1);  
//   cv::drawContours(mask_img, contours, -1, CV_RGB(255, 255, 255), CV_FILLED);
//   imshow("line_img",line_img);
//   imshow("mask_img",mask_img);
//   waitKey(0);
  /***to map X-x***/
    if(contours.size()==0){
        return;
    }
  std::vector<cv::Point> &contour=contours[0];
  int near[9][2]={{0,0},{0,-1},{0,1},{-1,0},{1,0},{1,1},{1,-1},{-1,1},{-1,-1}};
  LOG(INFO)<<"contour.size() : "<<contour.size();
  Mat extrinsic =Transformation::getTransformationMatrix(pose);
  for (int i = 0; i < contour.size(); ++i){
      for(int j=0;j<9;j++){
// 	cv::Point cpt = contour[i];      
	int value = img1.at<int>(contour[i].y+near[j][0],contour[i].x+near[j][1]);
	if (value > 0) {
	  img1.at<int>(contour[i].y+near[j][0],contour[i].x+near[j][1])=0;
// 	  LOG(INFO)<<" value: "<<value;
	  cv::Point3d pt3d( visible_Xs.at<float>(0, value - 1), visible_Xs.at<float>(1, value - 1),visible_Xs.at<float>(2, value - 1));	  
	  contour_Xs.push_back(pt3d);
	  contour_xs.push_back(X_to_x(pt3d,extrinsic));
	  break;
	}  
      }
  }
  LOG(INFO)<<"contour_Xs.size = "<<contour_Xs.size();
}
void Model::getContourPointsAndIts3DPoints(const  float *pose,float(*ctrPts3DMem)[3],float(*ctrPts2DMem)[2],int &nctrPts)
{
  std::vector<cv::Point3d> contour_Xs;
  std::vector<cv::Point2d> contour_xs;
  getContourPointsAndIts3DPoints(pose,contour_Xs,contour_xs);
  nctrPts=contour_Xs.size();
  ctrPts3DMem = (float(*)[3])malloc(nctrPts*sizeof(float[3]));
  ctrPts2DMem = (float(*)[2])malloc(nctrPts*sizeof(float[2]));
  for(int i=0;i<contour_Xs.size();i++){
    ctrPts3DMem[i][0]=contour_Xs[i].x;
    ctrPts3DMem[i][1]=contour_Xs[i].y;
    ctrPts3DMem[i][2]=contour_Xs[i].z;
    ctrPts2DMem[i][0]=contour_xs[i].x;
    ctrPts2DMem[i][1]=contour_xs[i].y;
  }
  
}


bool Model::pointInFrame(const Point& pt)
{
  return(pt.x>=0&&pt.y>=0&&pt.x<m_width&&pt.y<m_height);
}



void Model::getIntrinsic(cv::Mat& intrinsic) const
{
  intrinsic = m_calibration.getIntrinsic();
}

