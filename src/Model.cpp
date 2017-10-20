#include <iostream>
#include "ObjectDetector/Model.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>
#include <opencv/cv.h>
#include <vector>
#include "ObjectDetector/Render.h"
#include <omp.h>

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

void Model::generatePoints()
{
 float step_X=0.01;
 for(int i=0; i<m_model->numLines; i++)
 {
   Line newLine;
   newLine.e1=m_model->lines[i].vindices[0];
   newLine.e2=m_model->lines[i].vindices[1];
   Vec3f p1=getPos_E(newLine.e1);
   Vec3f p2=getPos_E(newLine.e2);
   
   newLine.points.push_back(p1);
   newLine.points.push_back(p2);
   
   Vec3f dx=(p2-p1);
   int Nx=sqrt(dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2])/step_X;
   dx /=Nx;
   Vec3f X=p1;
   
   for(int i=1;i<=Nx;++i,X+=dx){
      newLine.points.push_back(X);
   }
   myLines.push_back(newLine);
 }
}
const vector< Model::Line >& Model::getMyLines()
{
  return myLines;
}

void Model::GetImagePoints(const cv::Mat& pose, PointSet& pointset)
{
	//("pose",pose);
	pointset.m_img_points.clear();
	pointset.m_img_points_f.clear();
	float x = pose.at<float>(0,0); float y = pose.at<float>(0,1); float z = pose.at<float>(0,2);
	float rx = pose.at<float>(0,3); float ry = pose.at<float>(0,4); float rz = pose.at<float>(0,5);


	
	cv::Mat extrinsic = GetPoseMatrix(pose);

	
	cv::Mat pos(4,m_model->numvertices,CV_32FC1);
	cv::Mat result(3,m_model->numvertices,CV_32FC1);
	for(int i=1; i<=m_model->numvertices; i++)
	{
	  pos.at<float>(0,i-1) = m_model->vertices[3*(i)+0];
	  pos.at<float>(1,i-1) = m_model->vertices[3*(i)+1];
	  pos.at<float>(2,i-1) = m_model->vertices[3*(i)+2];
	  pos.at<float>(3,i-1) = 1;
	}
	
	
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
float Model::crossProductNorm(const Point2f p, const Point2f p1)
{
      return p.x*p1.y - p.y*p1.x;

}

void Model::checkPointInTrinangle(const cv::Point2f p,const cv::Point2f a, const cv::Point2f b,const cv::Point2f c){
  cv::Point2f pa=p-a,
	      pb=p-b,
	      pc=p-c;
  float t1 = crossProductNorm(pa,pb);
  float t2 = crossProductNorm(pa,pc);
  float t3 = crossProductNorm(pc,pb);
  return t1*t2 >= 0 && t1*t3 >= 0;
}
bool Model::isLineVisible(const Point2f& v1, const Point2f& v2, const PointSet& point_set)
{
   for (int j = 0; j < m_model->numtriangles; j++) {
      cv::Point2f p1= point_set.m_img_points_f[m_model->triangles->vindices[0]];
      cv::Point2f p2= point_set.m_img_points_f[m_model->triangles->vindices[1]];
      cv::Point2f p3= point_set.m_img_points_f[m_model->triangles->vindices[2]]; 
    return (1-
      checkPointInTrinangle(cv::Point2f(v1.x,v1.y+2),p1,p2,p3)&&
      checkPointInTrinangle(cv::Point2f(v1.x,v1.y-2),p1,p2,p3)&&
      checkPointInTrinangle(cv::Point2f(v1.x+2,v1.y),p1,p2,p3)&&
      checkPointInTrinangle(cv::Point2f(v1.x-2,v1.y),p1,p2,p3)&&
      checkPointInTrinangle(cv::Point2f(v2.x,v2.y+2),p1,p2,p3)&&
      checkPointInTrinangle(cv::Point2f(v2.x,v2.y-2),p1,p2,p3)&&
      checkPointInTrinangle(cv::Point2f(v2.x+2,v2.y),p1,p2,p3)&&
      checkPointInTrinangle(cv::Point2f(v2.x-2,v2.y),p1,p2,p3));
   }
}



void Model::visibleLinesAtPose(const Mat pose)
{
  cv::Mat extrinsic = GetPoseMatrix(pose);
  cv::Mat pos(4,m_model->numvertices,CV_32FC1);
  PointSet pointset;
  GetImagePoints(pose,pointset);
  for(int i=0;i<m_model->numLines.size();i++){
    if(m_model->lines[i].visible){
      cv::Point2f v1=pointset.m_img_points_f[m_model->lines[i].vindices[0]];
      cv::Point2f v2=pointset.m_img_points_f[m_model->lines[i].vindices[1]];
      m_model->lines[i].visible=isLineVisible(v1,v2,pointset);
    }
  }
  
}

void Model::DisplayCV(const cv::Mat& pose, cv::Mat& frame)
{
	float x = pose.at<float>(0,0); float y = pose.at<float>(0,1); float z = pose.at<float>(0,2);
	float rx = pose.at<float>(0,3); float ry = pose.at<float>(0,4); float rz = pose.at<float>(0,5);


	//camera intrinsic
	cv::Mat intrinsic(3,4,CV_32FC1);
	intrinsic.at<float>(0,0)=m_calibration.fx(); intrinsic.at<float>(0,1)=0; intrinsic.at<float>(0,2)=m_calibration.cx(); intrinsic.at<float>(0,3)=0;
	intrinsic.at<float>(1,0)=0; intrinsic.at<float>(1,1)=m_calibration.fy(); intrinsic.at<float>(1,2)=m_calibration.cy(); intrinsic.at<float>(1,3)=0;
	intrinsic.at<float>(2,0)=0; intrinsic.at<float>(2,1)=0; intrinsic.at<float>(2,2)=1; intrinsic.at<float>(2,3)=0;

	//camera extrinsic
	cv::Mat extrinsic = GetPoseMatrix(pose);

	//extract the points which can be visible or in the edge of object
	std::vector<cv::Point> vertexIndexs;
	std::vector<cv::Point3f> points_3d;

	for(int i=0; i<m_model->numLines; i++)
	{
// 		if((m_model->lines[i].e1 == 1 && m_model->lines[i].e2 == 0) || (m_model->lines[i].e1==0 && m_model->lines[i].e2==1) || (m_model->lines[i].e1==1 && m_model->lines[i].e2==1))
// 	  if(!isSameNormal(m_model->lines[i].n1, m_model->lines[i].n2)){
	  if(m_model->lines[i].visible){
		GLuint v0 = m_model->lines[i].vindices[0];
		GLuint v1 = m_model->lines[i].vindices[1];

		float x0 = m_model->vertices[3*v0];
		float y0 = m_model->vertices[3*v0+1];
		float z0 = m_model->vertices[3*v0+2];

		float x1 = m_model->vertices[3*v1];
		float y1 = m_model->vertices[3*v1+1];
		float z1 = m_model->vertices[3*v1+2];

		points_3d.push_back(cv::Point3f(x0,y0,z0));
		points_3d.push_back(cv::Point3f(x1,y1,z1));
			
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

	//display the visible lines
	for(int i=0; i<vertexIndexs.size(); i++)
	{
		int u1 = result.at<float>(0,2*i)/result.at<float>(2,2*i);
		int v1 = result.at<float>(1,2*i)/result.at<float>(2,2*i);
		
		int u2 = result.at<float>(0,2*i+1)/result.at<float>(2,2*i+1);
		int v2 = result.at<float>(1,2*i+1)/result.at<float>(2,2*i+1);
		if(u1 >=0 && u1<frame.cols && v1 >=0 && v1<=frame.rows && u2 >=0 && u2<frame.cols && v2>=0 && v2<=frame.rows)
		{
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
cv::Vec3f Model::getPos_E(int e)
{
  float x0 = m_model->vertices[3*e];
  float y0 = m_model->vertices[3*e+1];
  float z0 = m_model->vertices[3*e+2];
  return cv::Vec3f(x0,y0,z0);
}
bool Model::isSameNormal(const float* n1, const float* n2)
{
  int k=1;
  float n[3];
  glmCross((GLfloat *)n1,(GLfloat *)n2,n);
  return (abs(n[0])<1e-2&&abs(n[1])<1e-2&&abs(n[2])<1e-2);
  
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
Mat Model::GetPoseMatrix(cv::Mat pose)
{
  cv::Mat roV(3,1,CV_32FC1);
  roV.at<float>(0,0) = pose.at<float>(0,0); 
  roV.at<float>(1,0) = pose.at<float>(0,1); 
  roV.at<float>(2,0) = pose.at<float>(0,2);
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
  T.at<float>(0,3)=pose.at<float>(0,3); T.at<float>(1,3)=pose.at<float>(0,4); T.at<float>(2,3) = pose.at<float>(0,5);
  return T;
}

void Model::getIntrinsic(cv::Mat& intrinsic) const
{
	intrinsic = m_calibration.getIntrinsic();

}

void Model::InitPose(const cv::Mat& initPose)
{

  m_rvec.at<float>(0,0) = initPose.at<float>(0,0); m_rvec.at<float>(1,0) = initPose.at<float>(0,1); m_rvec.at<float>(2,0) = initPose.at<float>(0,2);
  m_tvec.at<float>(0,0) = initPose.at<float>(0,3); m_tvec.at<float>(1,0) = initPose.at<float>(0,4); m_tvec.at<float>(2,0) = initPose.at<float>(0,5);
}