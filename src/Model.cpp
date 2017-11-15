#include <iostream>
#include "ObjectDetector/Model.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>
#include <opencv/cv.h>
#include <vector>
#include "ObjectDetector/Render.h"
#include "ObjectDetector/Transformation.h"
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
Model::Model(const Config& config) : m_bb_points(4,8,CV_32FC1)
{
	m_model = config.model;
	m_radius = glmMaxRadius(m_model);

	m_calibration = config.camCalibration;
	m_width = config.width;
	m_height = config.height;
	m_rvec.create(3,1,CV_32FC1);
	m_tvec.create(3,1,CV_32FC1);
	getVisibleLines();
	
	intrinsic=cv::Mat(3,4,CV_32FC1);
	intrinsic.at<float>(0,0)=m_calibration.fx(); intrinsic.at<float>(0,1)=0; intrinsic.at<float>(0,2)=m_calibration.cx(); intrinsic.at<float>(0,3)=0;
	intrinsic.at<float>(1,0)=0; intrinsic.at<float>(1,1)=m_calibration.fy(); intrinsic.at<float>(1,2)=m_calibration.cy(); intrinsic.at<float>(1,3)=0;
	intrinsic.at<float>(2,0)=0; intrinsic.at<float>(2,1)=0; intrinsic.at<float>(2,2)=1; intrinsic.at<float>(2,3)=0;
	
	pos =cv::Mat (4,m_model->numvertices,CV_32FC1);
	for(int i=1; i<=m_model->numvertices; i++)
	{
	  pos.at<float>(0,i-1) = m_model->vertices[3*(i)+0];
	  pos.at<float>(1,i-1) = m_model->vertices[3*(i)+1];
	  pos.at<float>(2,i-1) = m_model->vertices[3*(i)+2];
	  pos.at<float>(3,i-1) = 1;
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

const vector< Line >& Model::getMyLines()
{
  return myLines;
}
void Model::GetImagePoints(const float* prepose, PointSet& pointset)
{
	pointset.m_img_points.clear();
	pointset.m_img_points_f.clear();	
	cv::Mat extrinsic = GetPoseMatrix(prepose);
	
	cv::Mat result(3,m_model->numvertices,CV_32FC1);

	result = intrinsic*extrinsic*pos;
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

bool Model::checkPointInTrinangle(const cv::Point p,const cv::Point a, const cv::Point b,const cv::Point c){
  cv::Point pa=a-p,
	    pb=b-p,
	    pc=c-p;
 
  int t1 = crossProductNorm(pa,pb);
  int t2 = crossProductNorm(pb,pc);
  int t3 = crossProductNorm(pc,pa);
 
 
  return t1*t2 >= 0 && t1*t3 >= 0;
}
bool Model::isLineVisible(const Point& v1, const Point& v2, const PointSet& point_set)
{
  return isPointVisible(v1,point_set)&&isPointVisible(v2,point_set);
}

bool Model::isPointVisible(const Point& vertice, const PointSet& point_set)
{
    float move=1;
    int four[4]={0};
    for (int j = 0; j < m_model->numtriangles; j++) {
      cv::Point p1= point_set.m_img_points[m_model->triangles->vindices[0]-1];
      cv::Point p2= point_set.m_img_points[m_model->triangles->vindices[1]-1];
      cv::Point p3= point_set.m_img_points[m_model->triangles->vindices[2]-1]; 
//       if(checkPointInTrinangle(cv::Point(vertice.x,vertice.y-move),p1,p2,p3)){
// 	printf("in %d %d %d",m_model->triangles->vindices[0],m_model->triangles->vindices[1],m_model->triangles->vindices[2]);
//       }
      four[0]+=checkPointInTrinangle(cv::Point(vertice.x,vertice.y+move),p1,p2,p3);
    }
//     printf("in %d \n",four[0]);
    if( (four[0]==0||four[1]==0||four[2]==0||four[3]==0)){
      return true;
    }
//     return false;
}
const cv::Mat& Model::getIntrinsic() const
{
  return intrinsic;
}
void Model::setVisibleLinesAtPose(const float * pose)
{
  cv::Mat extrinsic = Transformation::getTransformationMatrix(pose);
  cv::Mat pos(4,m_model->numvertices,CV_32FC1);
  PointSet pointset;
  GetImagePoints(pose,pointset);
  for(int i=0;i<m_model->numLines;i++){
    if(m_model->lines[i].visible){
      cv::Point v1=pointset.m_img_points[m_model->lines[i].vindices[0]-1];
      cv::Point v2=pointset.m_img_points[m_model->lines[i].vindices[1]-1];

      m_model->lines[i].tovisit=isLineVisible(v1,v2,pointset);

      if((m_model->lines[i].vindices[0]==8&&m_model->lines[i].vindices[1]==4) ||m_model->lines[i].vindices[0]==5||m_model->lines[i].vindices[1]==5){
	m_model->lines[i].tovisit=false;
      }
    }else{
      m_model->lines[i].tovisit=false;
    }
  }
  
}
Point Model::X_to_x(Point3f X,Mat extrisic)
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

void Model::DisplayLine(const cv::Point& p1,const cv::Point& p2, cv::Mat& frame)
{
//   int u1=p1.y,v1=p1.x,v2=p2.y,u2=p2.x;
/*  if(u1 >=0 && u1<frame.cols && v1 >=0 && v1<=frame.rows && u2 >=0 && u2<frame.cols && v2>=0 && v2<=frame.rows)
  {
    LOG(INFO)<< "v0:"<< u1<<" "<<v1<< " , v1:"<< u2<<" "<<v2<<std::endl;*/ 
    cv::circle(frame,p1,10,Scalar(0,0,255));  

    cv::line(frame,p1,p2,cv::Scalar(0,255,0),1,CV_AA);
//   }
}
void Model::DisplayCV(const float * pose, cv::Mat& frame)
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
		
		
		Point3f p1= Point3f(pos.at<float>(0,v0-1),pos.at<float>(1,v0-1),pos.at<float>(2,v0-1));
		Point3f p2= Point3f(pos.at<float>(0,v1-1),pos.at<float>(1,v1-1),pos.at<float>(2,v1-1));

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
	LOG(INFO)<<"----2D coordinate of tovisit line";
	//display the visible lines
	for(int i=0; i<vertexIndexs.size(); i++)
	{
		int u1 = result.at<float>(0,2*i)/result.at<float>(2,2*i);
		int v1 = result.at<float>(1,2*i)/result.at<float>(2,2*i);
		
		int u2 = result.at<float>(0,2*i+1)/result.at<float>(2,2*i+1);
		int v2 = result.at<float>(1,2*i+1)/result.at<float>(2,2*i+1);
				 
		LOG(INFO)<<"frame.cols = "<<frame.cols<<" , frame.rows= "<<frame.rows; 

		if(u1 >=0 && u1<frame.cols && v1 >=0 && v1<=frame.rows && u2 >=0 && u2<frame.cols && v2>=0 && v2<=frame.rows)
		{
		  LOG(INFO)<< "v0:"<< u1<<" "<<v1<< " , v1:"<< u2<<" "<<v2<<std::endl;

		  cv::line(frame,cv::Point(u1,v1),cv::Point(u2,v2),cv::Scalar(0,0,255),1,CV_AA);
		}
	}
}

void Model::DisplayGL(const cv::Mat& prepose)
{
	//compute extrinsic
	g_render.m_shapePoseInfo.clear();
	float x = prepose.at<float>(0,0); float y = prepose.at<float>(0,1); float z = prepose.at<float>(0,2);
	float rx = prepose.at<float>(0,3); float ry = prepose.at<float>(0,4); float rz = prepose.at<float>(0,5);

#ifdef MY
	//camera extrinsic
	cv::Mat extrinsic(4,4,CV_32FC1);
	const float pi = 3.1415926f;
	extrinsic.at<float>(0,0)=cos(ry)*cos(rz); extrinsic.at<float>(0,1)=sin(rx)*sin(ry)-cos(rx)*cos(ry)*sin(rz); extrinsic.at<float>(0,2)=cos(rx)*sin(ry)+cos(ry)*sin(rx)*sin(rz); extrinsic.at<float>(0,3)=x;
	extrinsic.at<float>(1,0)=sin(rz); extrinsic.at<float>(1,1)=cos(rx)*cos(rz); extrinsic.at<float>(1,2)=-cos(rz)*sin(rx); extrinsic.at<float>(1,3)=y;
	extrinsic.at<float>(2,0)=-cos(rz)*sin(ry); extrinsic.at<float>(2,1)=cos(ry)*sin(rx)+cos(rx)*sin(ry)*sin(rz); extrinsic.at<float>(2,2)=cos(rx)*cos(ry)-sin(rx)*sin(ry)*sin(rz); extrinsic.at<float>(2,3)=z;
	extrinsic.at<float>(3,0)=0; extrinsic.at<float>(3,1)=0; extrinsic.at<float>(3,2)=0; extrinsic.at<float>(3,3)=1;

#else
	cv::Mat extrinsic = GetPoseMatrix();
#endif

	ORD::ShapePoseInfo shapePoseInfo;
	shapePoseInfo.m_shape = m_model;
	g_render.matrixFromCV2GL(extrinsic,shapePoseInfo.mv_matrix);
	g_render.m_shapePoseInfo.push_back(shapePoseInfo);
	g_render.rendering();

	//cv::Mat tmp = g_render.getRenderedImg();
	////change to 3 channel
	//cv::Mat render_img(tmp.rows,tmp.cols,CV_8UC3);
	//for(int r=0; r<tmp.rows; r++)
	//	for(int c=0; c<tmp.cols; c++)
	//	{
	//		render_img.at<cv::Vec3b>(r,c)[0] = tmp.at<uchar>(r,c);
	//		render_img.at<cv::Vec3b>(r,c)[1] = tmp.at<uchar>(r,c);
	//		render_img.at<cv::Vec3b>(r,c)[2] = tmp.at<uchar>(r,c);
	//	}
	//static cv::VideoWriter writer("result_s7_render.avi",CV_FOURCC('D','I','V','X'),30,render_img.size());
	//writer<<render_img;
}

static void Extrinsic(cv::Mat* rotMatrix, float rx, float ry, float rz)
{
	//ry*rz*rx
	const float pi = 3.1415926f;

	(*rotMatrix).at<float>(0,0)=cos(ry)*cos(rz); (*rotMatrix).at<float>(0,1)=sin(rx)*sin(ry)-cos(rx)*cos(ry)*sin(rz); (*rotMatrix).at<float>(0,2)=cos(rx)*sin(ry)+cos(ry)*sin(rx)*sin(rz);
	(*rotMatrix).at<float>(1,0)=sin(rz); (*rotMatrix).at<float>(1,1)=cos(rx)*cos(rz); (*rotMatrix).at<float>(1,2)=-cos(rz)*sin(rx);
	(*rotMatrix).at<float>(2,0)=-cos(rz)*sin(ry); (*rotMatrix).at<float>(2,1)=cos(ry)*sin(rx)+cos(rx)*sin(ry)*sin(rz); (*rotMatrix).at<float>(2,2)=cos(rx)*cos(ry)-sin(rx)*sin(ry)*sin(rz);

	//ry*rx*rz
	/*(*rotMatrix).at<float>(0,0)=cos(ry)*cos(rz)+sin(rx)*sin(ry)*sin(rz); (*rotMatrix).at<float>(0,1)=cos(rz)*sin(rx)*sin(ry)-cos(ry)*sin(rz); (*rotMatrix).at<float>(0,2)=cos(rx)*sin(ry);
	(*rotMatrix).at<float>(1,0)=cos(rx)*sin(rz); (*rotMatrix).at<float>(1,1)=cos(rx)*cos(rz); (*rotMatrix).at<float>(1,2)=-sin(rx);
	(*rotMatrix).at<float>(2,0)=cos(ry)*sin(rx)*sin(rz)-cos(rz)*sin(ry); (*rotMatrix).at<float>(2,1)=sin(ry)*sin(rz)+cos(ry)*cos(rz)*sin(rx); (*rotMatrix).at<float>(2,2)=cos(rx)*cos(ry);*/
}



void Model::computeExtrinsicByEuler(cv::Mat* mvMatrix, float& _x, float& _y, float& _z, float& _rx, float &_ry, float &_rz)
{
	const float pi = 3.1415926f;
	//float rx = _rx*pi/180; float ry = _ry*pi/180; float rz = _rz*pi/180;
	float rx = _rx; float ry = _ry; float rz = _rz;
	//openglÓëopencv×ø±ê²îÈÆx180¶È
	(*mvMatrix).at<float>(0,0)=cos(ry)*cos(rz); (*mvMatrix).at<float>(0,1)=sin(rx)*sin(ry)-cos(rx)*cos(ry)*sin(rz); (*mvMatrix).at<float>(0,2)=cos(rx)*sin(ry)+cos(ry)*sin(rx)*sin(rz); (*mvMatrix).at<float>(0,3)=_x;
	(*mvMatrix).at<float>(1,0)=sin(rz); (*mvMatrix).at<float>(1,1)=cos(rx)*cos(rz); (*mvMatrix).at<float>(1,2)=-cos(rz)*sin(rx); (*mvMatrix).at<float>(1,3)=_y;
	(*mvMatrix).at<float>(2,0)=-cos(rz)*sin(ry); (*mvMatrix).at<float>(2,1)=cos(ry)*sin(rx)+cos(rx)*sin(ry)*sin(rz); (*mvMatrix).at<float>(2,2)=cos(rx)*cos(ry)-sin(rx)*sin(ry)*sin(rz); (*mvMatrix).at<float>(2,3)=_z;
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
  return pos;
}

cv::Mat Model::GetPoseMatrix()
{
	cv::Mat rotMat(3,3,CV_32FC1);
	cv::Mat T = cv::Mat::eye(4,4,CV_32FC1);

	cv::Rodrigues(m_rvec,rotMat);
	for(int c=0; c<3; c++)
	{
		for(int r=0; r<3; r++)
		{
			T.at<float>(r,c) = rotMat.at<float>(r,c);
		}
	}
	T.at<float>(0,3)=m_tvec.at<float>(0,0); T.at<float>(1,3)=m_tvec.at<float>(1,0); T.at<float>(2,3) = m_tvec.at<float>(2,0);
	return T;
}
Mat Model::GetPoseMatrix(const float* pose)
{
  cv::Mat roV(3,1,CV_32FC1);
  roV.at<float>(0,0) = pose[0]; 
  roV.at<float>(1,0) = pose[1]; 
  roV.at<float>(2,0) = pose[2];
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
  T.at<float>(0,3)=pose[3]; T.at<float>(1,3)=pose[4]; T.at<float>(2,3) = pose[5];
  return T;
}


void Model::getIntrinsic(cv::Mat& intrinsic) const
{
  intrinsic = m_calibration.getIntrinsic();
}

