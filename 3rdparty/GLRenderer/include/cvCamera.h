#ifndef _CVCAMERA_H_
#define _CVCAMERA_H_

#include <opencv2/opencv.hpp>
#include <vector>

class Camera
{
public:
	Camera();
	Camera(float fx, float fy, float cx, float cy);
	Camera(float fx, float fy, float cx, float cy, float distortion[5]);
	Camera(const Camera& cam);
	~Camera();

	float getfx() const;
	float getfy() const;
	float getcx() const;
	float getcy() const;
	cv::Mat getIntrinsic() const;
	const float* getProjectionIntrinsic(float width, float height, float nearP, float farP);
	const float* getProjectionIntrinsicDefaultPrincipalPoint(float width, float height, float nearP, float farP);
	cv::Mat getExtrinsic() const;
	const float* getModelviewExtrinsic();
	cv::Mat getDistorsions() const;

	void setIntrinsic(float fx, float fy, float cx, float cy);
	void setIntrinsic(const cv::Mat& intrinsic);
	void setExtrinsic(float rx, float ry, float rz, float tx, float ty, float tz); // Rodrigues rotation
	void setExtrinsic(const cv::Mat& extrinsic);
	void setDistortion(float distortion[5]);
	void setDistortion(const cv::Mat& distortion);

	// convert the camera coordinate system between opencv and opengl context
	cv::Mat flipExtrinsicYZ() const;

	void copyFrom(const Camera& cam);

	// get/set R and t (the pose the target object)
	void getRt(cv::Mat &R, cv::Mat &t) const;
	void setRt(const cv::Mat &R, const cv::Mat &t);
	void setRt(float ex, float ey, float ez, float tx, float ty, float tz); // euler angle in XZY format

private:
	cv::Mat m_intrinsic; // camera intrinsic parameters 3*3
	cv::Mat m_distortion; // camera distortion parameters 5*1
	cv::Mat m_extrinsic; // camera extrinsic parameters 3*4

	float m_fx, m_fy, m_cx, m_cy;

	float *glProjectMatrix;
	float *glModelviewMatrix;
};

#endif