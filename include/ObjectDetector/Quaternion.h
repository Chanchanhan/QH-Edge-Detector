#pragma once

#include<opencv2/highgui.hpp>
namespace OD{
class Quaternion
{
public:
	float x , y , z , w;
public:
	Quaternion(void) : x(0.0f) , y(0.0f) , z(0.0f) , w(1.0f) {}
	~Quaternion(void) {}
	
	
	void Normalize();
	void SetEulerAngle(const cv::Vec3f& ea);
	Quaternion dAxB(const float dA0,const float dA1,const float dA2);
	Quaternion dAxB(const cv::Vec3f &dA);
	void SetRotVec(const cv::Vec3f &rvec);
	cv::Vec3f GetEulerAngle() const;
	cv::Vec3f GetRotVec() const;
};
}