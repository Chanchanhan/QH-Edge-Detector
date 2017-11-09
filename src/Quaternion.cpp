#include "ObjectDetector/Quaternion.h"
#include <math.h>

#define CLAMP(x , min , max) ((x) > (max) ? (max) : ((x) < (min) ? (min) : x))
using namespace OD;
void Quaternion::SetEulerAngle(const cv::Vec3f &ea)
{
	float fCosHRoll = cos(ea[0] * .5f);
	float fSinHRoll = sin(ea[0] * .5f);
	float fCosHPitch = cos(ea[1] * .5f);
	float fSinHPitch = sin(ea[1] * .5f);
	float fCosHYaw = cos(ea[2] * .5f);
	float fSinHYaw = sin(ea[2] * .5f);

	/// Cartesian coordinate System
	//w = fCosHRoll * fCosHPitch * fCosHYaw + fSinHRoll * fSinHPitch * fSinHYaw;
	//x = fSinHRoll * fCosHPitch * fCosHYaw - fCosHRoll * fSinHPitch * fSinHYaw;
	//y = fCosHRoll * fSinHPitch * fCosHYaw + fSinHRoll * fCosHPitch * fSinHYaw;
	//z = fCosHRoll * fCosHPitch * fSinHYaw - fSinHRoll * fSinHPitch * fCosHYaw;

	w = fCosHRoll * fCosHPitch * fCosHYaw + fSinHRoll * fSinHPitch * fSinHYaw;
	x = fCosHRoll * fSinHPitch * fCosHYaw + fSinHRoll * fCosHPitch * fSinHYaw;
	y = fCosHRoll * fCosHPitch * fSinHYaw - fSinHRoll * fSinHPitch * fCosHYaw;
	z = fSinHRoll * fCosHPitch * fCosHYaw - fCosHRoll * fSinHPitch * fSinHYaw;
	Normalize();
}
cv::Vec3f Quaternion::GetRotVec() const
{
  cv::Vec3f axis(x,y,z),rvec;
  float theta = acos(z) * 2;  
  
  axis =axis/sin(theta/2);  
  axis = axis /cv::norm(axis);  
  
  rvec = axis*theta;  
  return rvec;
}

cv::Vec3f Quaternion::GetEulerAngle() const
{
	cv::Vec3f ea;

	/// Cartesian coordinate System 
	//ea.m_fRoll  = atan2(2 * (w * x + y * z) , 1 - 2 * (x * x + y * y));
	//ea.m_fPitch = asin(2 * (w * y - z * x));
	//ea.m_fYaw   = atan2(2 * (w * z + x * y) , 1 - 2 * (y * y + z * z));

	ea[0] = atan2(2 * (w * z + x * y) , 1 - 2 * (z * z + x * x));
	ea[1] = asin(CLAMP(2 * (w * x - y * z) , -1.0f , 1.0f));
	ea[2] = atan2(2 * (w * y + z * x) , 1 - 2 * (x * x + y * y));

	return ea;
}
Quaternion Quaternion::dAxB(const float dA0,const float dA1,const float dA2)
{
  Quaternion dAxB;
  dAxB.x = ( dA2 * y - dA1 * z + dA0 * w) * 0.5f + x;
  dAxB.y = (-dA2 * x + dA0 * z + dA1 * w) * 0.5f + y;
  dAxB.z = ( dA1 * x - dA0 * y + dA2 * w) * 0.5f + z;
  dAxB.w = (-dA0 * x - dA1 * y - dA2 * z) * 0.5f + w;
  dAxB.Normalize();
  return dAxB;
}
Quaternion Quaternion::dAxB(const cv::Vec3f &dA)
{
  Quaternion dAxB;
  dAxB.x = ( dA[2] * y - dA[1] * z + dA[0] * w) * 0.5f + x;
  dAxB.y = (-dA[2] * x + dA[0] * z + dA[1] * w) * 0.5f + y;
  dAxB.z = ( dA[1] * x - dA[0] * y + dA[2] * w) * 0.5f + z;
  dAxB.w = (-dA[0] * x - dA[1] * y - dA[2] * z) * 0.5f + w;
  dAxB.Normalize();
  return dAxB;
}
void Quaternion::Normalize()
{
  float S= sqrt(x*x+y*y+z*z+w*w);
  w= w<0? -w:w;
  x/=S;
  y/=S;
  z/=S;
  w/=S;  
}
