#include "../include/cvCamera.h"


Camera::Camera()
	: m_intrinsic(3, 3, CV_32FC1), m_extrinsic(3, 4, CV_32FC1), m_distortion(5, 1, CV_32FC1),
	m_fx(0.0f), m_fy(0.0f), m_cx(0.0f), m_cy(0.0f)
{
	m_intrinsic = cv::Mat::zeros(3, 3, CV_32FC1);
	m_extrinsic = cv::Mat::zeros(3, 4, CV_32FC1);
	m_distortion = cv::Mat::zeros(5, 1, CV_32FC1);

	glProjectMatrix = (float*)malloc(16 * sizeof(float));
	glModelviewMatrix = (float*)malloc(16 * sizeof(float));
}

Camera::Camera(float fx, float fy, float cx, float cy)
	: m_intrinsic(3, 3, CV_32FC1), m_extrinsic(3, 4, CV_32FC1), m_distortion(5, 1, CV_32FC1),
	m_fx(fx), m_fy(fy), m_cx(cx), m_cy(cy)
{
	m_intrinsic = cv::Mat::zeros(3, 3, CV_32FC1);
	m_extrinsic = cv::Mat::zeros(3, 4, CV_32FC1);
	m_distortion = cv::Mat::zeros(5, 1, CV_32FC1);

	m_intrinsic.ptr<float>(0)[0] = fx; m_intrinsic.ptr<float>(0)[1] = 0.0f; m_intrinsic.ptr<float>(0)[2] = cx;
	m_intrinsic.ptr<float>(1)[0] = 0.0f; m_intrinsic.ptr<float>(1)[1] = fy; m_intrinsic.ptr<float>(1)[2] = cy;
	m_intrinsic.ptr<float>(2)[0] = 0.0f; m_intrinsic.ptr<float>(2)[1] = 0.0f; m_intrinsic.ptr<float>(2)[2] = 1.0f;

	glProjectMatrix = (float*)malloc(16 * sizeof(float));
	glModelviewMatrix = (float*)malloc(16 * sizeof(float));
}

Camera::Camera(float fx, float fy, float cx, float cy, float distortion[5])
	: m_intrinsic(3, 3, CV_32FC1), m_extrinsic(3, 4, CV_32FC1), m_distortion(5, 1, CV_32FC1),
	m_fx(fx), m_fy(fy), m_cx(cx), m_cy(cy)
{
	m_intrinsic = cv::Mat::zeros(3, 3, CV_32FC1);
	m_extrinsic = cv::Mat::zeros(3, 4, CV_32FC1);
	m_distortion = cv::Mat::zeros(5, 1, CV_32FC1);

	m_intrinsic.ptr<float>(0)[0] = fx; m_intrinsic.ptr<float>(0)[1] = 0.0f; m_intrinsic.ptr<float>(0)[2] = cx;
	m_intrinsic.ptr<float>(1)[0] = 0.0f; m_intrinsic.ptr<float>(1)[1] = fy; m_intrinsic.ptr<float>(1)[2] = cy;
	m_intrinsic.ptr<float>(2)[0] = 0.0f; m_intrinsic.ptr<float>(2)[1] = 0.0f; m_intrinsic.ptr<float>(2)[2] = 1.0f;

	m_distortion.ptr<float>(0)[0] = distortion[0]; m_distortion.ptr<float>(1)[0] = distortion[1];
	m_distortion.ptr<float>(2)[0] = distortion[2]; m_distortion.ptr<float>(3)[0] = distortion[3];
	m_distortion.ptr<float>(4)[0] = distortion[4];

	glProjectMatrix = (float*)malloc(16 * sizeof(float));
	glModelviewMatrix = (float*)malloc(16 * sizeof(float));
}

Camera::Camera(const Camera& cam)
{
	m_intrinsic = cam.m_intrinsic.clone();
	m_extrinsic = cam.m_extrinsic.clone();
	m_distortion = cam.m_distortion.clone();

	m_fx = cam.m_fx;
	m_fy = cam.m_fy;
	m_cx = cam.m_cx;
	m_cy = cam.m_cy;

	glProjectMatrix = (float*)malloc(16 * sizeof(float));
	glModelviewMatrix = (float*)malloc(16 * sizeof(float));

	memcpy(glProjectMatrix, cam.glModelviewMatrix, 16 * sizeof(float));
	memcpy(glModelviewMatrix, cam.glModelviewMatrix, 16 * sizeof(float));
}

void Camera::copyFrom(const Camera& cam)
{
	m_intrinsic = cam.m_intrinsic.clone();
	m_extrinsic = cam.m_extrinsic.clone();
	m_distortion = cam.m_distortion.clone();

	m_fx = cam.m_fx;
	m_fy = cam.m_fy;
	m_cx = cam.m_cx;
	m_cy = cam.m_cy;

	memcpy(glProjectMatrix, cam.glProjectMatrix, 16 * sizeof(float));
	memcpy(glModelviewMatrix, cam.glModelviewMatrix, 16 * sizeof(float));
}

Camera::~Camera()
{
	free(glProjectMatrix);
	free(glModelviewMatrix);
}

float Camera::getfx() const
{
	return m_fx;
}

float Camera::getfy() const
{
	return m_fy;
}

float Camera::getcx() const
{
	return m_cx;
}

float Camera::getcy() const
{
	return m_cy;
}

cv::Mat Camera::getIntrinsic() const
{
	return m_intrinsic;
}

const float* Camera::getProjectionIntrinsic(float width, float height, float nearP, float farP)
{
	// in opengl context, the matrix is stored in column-major order 
	glProjectMatrix[0] = 2.0f * m_fx / width;
	glProjectMatrix[1] = 0.0f;
	glProjectMatrix[2] = 0.0f;
	glProjectMatrix[3] = 0.0f;

	glProjectMatrix[4] = 0.0f;
	glProjectMatrix[5] = 2.0f * m_fy / height;
	glProjectMatrix[6] = 0.0f;
	glProjectMatrix[7] = 0.0f;


	glProjectMatrix[8] = 2.0f * m_cx / width - 1.0f;
	glProjectMatrix[9] = 2.0f * m_cy / height - 1.0f;
	glProjectMatrix[10] = -(farP + nearP) / (farP - nearP);
	glProjectMatrix[11] = -1.0f;


	glProjectMatrix[12] = 0.0f;
	glProjectMatrix[13] = 0.0f;
	glProjectMatrix[14] = -2.0f * farP * nearP / (farP - nearP);
	glProjectMatrix[15] = 0.0f;

	return glProjectMatrix;
}

const float* Camera::getProjectionIntrinsicDefaultPrincipalPoint(float width, float height, float nearP, float farP)
{
	// in opengl context, the matrix is stored in column-major order 
	glProjectMatrix[0] = 2.0f * m_fx / width;
	glProjectMatrix[1] = 0.0f;
	glProjectMatrix[2] = 0.0f;
	glProjectMatrix[3] = 0.0f;

	glProjectMatrix[4] = 0.0f;
	glProjectMatrix[5] = 2.0f * m_fy / height;
	glProjectMatrix[6] = 0.0f;
	glProjectMatrix[7] = 0.0f;


	glProjectMatrix[8] = 0.0f;
	glProjectMatrix[9] = 0.0f;
	glProjectMatrix[10] = -(farP + nearP) / (farP - nearP);
	glProjectMatrix[11] = -1.0f;


	glProjectMatrix[12] = 0.0f;
	glProjectMatrix[13] = 0.0f;
	glProjectMatrix[14] = -2.0f * farP * nearP / (farP - nearP);
	glProjectMatrix[15] = 0.0f;

	return glProjectMatrix;
}

cv::Mat Camera::getExtrinsic() const
{
	return m_extrinsic;
}

const float* Camera::getModelviewExtrinsic()
{
	// from opencv context to opengl context
	// we need rotate the object frame along X with 180 degrees
	cv::Mat flipedExtrinsic = flipExtrinsicYZ();

	// in opengl context, the matrix is stored in column-major order 
	glModelviewMatrix[0] = flipedExtrinsic.ptr<float>(0)[0];
	glModelviewMatrix[1] = flipedExtrinsic.ptr<float>(1)[0];
	glModelviewMatrix[2] = flipedExtrinsic.ptr<float>(2)[0];
	glModelviewMatrix[3] = 0.0f;

	glModelviewMatrix[4] = flipedExtrinsic.ptr<float>(0)[1];
	glModelviewMatrix[5] = flipedExtrinsic.ptr<float>(1)[1];
	glModelviewMatrix[6] = flipedExtrinsic.ptr<float>(2)[1];
	glModelviewMatrix[7] = 0.0f;


	glModelviewMatrix[8] = flipedExtrinsic.ptr<float>(0)[2];
	glModelviewMatrix[9] = flipedExtrinsic.ptr<float>(1)[2];
	glModelviewMatrix[10] = flipedExtrinsic.ptr<float>(2)[2];
	glModelviewMatrix[11] = 0.0f;


	glModelviewMatrix[12] = flipedExtrinsic.ptr<float>(0)[3];
	glModelviewMatrix[13] = flipedExtrinsic.ptr<float>(1)[3];
	glModelviewMatrix[14] = flipedExtrinsic.ptr<float>(2)[3];
	glModelviewMatrix[15] = 1.0f;

	return glModelviewMatrix;
}

cv::Mat Camera::getDistorsions() const
{
	return m_distortion;
}

void Camera::setIntrinsic(float fx, float fy, float cx, float cy)
{
	m_fx = fx;
	m_fy = fy;
	m_cx = cx;
	m_cy = cy;
	m_intrinsic.ptr<float>(0)[0] = fx; m_intrinsic.ptr<float>(0)[1] = 0.0f; m_intrinsic.ptr<float>(0)[2] = cx;
	m_intrinsic.ptr<float>(1)[0] = 0.0f; m_intrinsic.ptr<float>(1)[1] = fy; m_intrinsic.ptr<float>(1)[2] = cy;
	m_intrinsic.ptr<float>(2)[0] = 0.0f; m_intrinsic.ptr<float>(2)[1] = 0.0f; m_intrinsic.ptr<float>(2)[2] = 1.0f;
}

void Camera::setIntrinsic(const cv::Mat& intrinsic)
{
	m_intrinsic = intrinsic.clone();
	m_fx = m_intrinsic.ptr<float>(0)[0];
	m_fy = m_intrinsic.ptr<float>(1)[1];
	m_cx = m_intrinsic.ptr<float>(0)[2];
	m_cy = m_intrinsic.ptr<float>(1)[2];
}

void Camera::setExtrinsic(float rx, float ry, float rz, float tx, float ty, float tz)
{
	cv::Mat rvmat = cv::Mat::zeros(3, 1, CV_32FC1);
	rvmat.ptr<float>(0)[0] = rx; rvmat.ptr<float>(1)[0] = ry; rvmat.ptr<float>(2)[0] = rz;
	cv::Mat rmat = cv::Mat::zeros(3, 3, CV_32FC1);
	cv::Rodrigues(rvmat, rmat);

	m_extrinsic.ptr<float>(0)[0] = rmat.ptr<float>(0)[0];
	m_extrinsic.ptr<float>(0)[1] = rmat.ptr<float>(0)[1];
	m_extrinsic.ptr<float>(0)[2] = rmat.ptr<float>(0)[2];
	m_extrinsic.ptr<float>(0)[3] = tx;

	m_extrinsic.ptr<float>(1)[0] = rmat.ptr<float>(1)[0];
	m_extrinsic.ptr<float>(1)[1] = rmat.ptr<float>(1)[1];
	m_extrinsic.ptr<float>(1)[2] = rmat.ptr<float>(1)[2];
	m_extrinsic.ptr<float>(1)[3] = ty;

	m_extrinsic.ptr<float>(2)[0] = rmat.ptr<float>(2)[0];
	m_extrinsic.ptr<float>(2)[1] = rmat.ptr<float>(2)[1];
	m_extrinsic.ptr<float>(2)[2] = rmat.ptr<float>(2)[2];
	m_extrinsic.ptr<float>(2)[3] = tz;
}

void Camera::setExtrinsic(const cv::Mat& extrinsic)
{
	m_extrinsic = extrinsic.clone();
}

void Camera::setDistortion(float distortion[5])
{
	m_distortion.ptr<float>(0)[0] = distortion[0]; m_distortion.ptr<float>(1)[0] = distortion[1];
	m_distortion.ptr<float>(2)[0] = distortion[2]; m_distortion.ptr<float>(3)[3] = distortion[3];
	m_distortion.ptr<float>(4)[0] = distortion[4];
}
void Camera::setDistortion(const cv::Mat& distortion)
{
	m_distortion = distortion.clone();
}

cv::Mat Camera::flipExtrinsicYZ() const
{
	// rotate along x with 180
	// the rotation matrix Rx is  
	// [1 0  0]
	// [0 -1 0]
	// [0 0 -1]
	// out = Rx x in
	cv::Mat flipedExtrinsic = m_extrinsic.clone();

	flipedExtrinsic.ptr<float>(1)[0] = -m_extrinsic.ptr<float>(1)[0];
	flipedExtrinsic.ptr<float>(1)[1] = -m_extrinsic.ptr<float>(1)[1];
	flipedExtrinsic.ptr<float>(1)[2] = -m_extrinsic.ptr<float>(1)[2];
	flipedExtrinsic.ptr<float>(1)[3] = -m_extrinsic.ptr<float>(1)[3];

	flipedExtrinsic.ptr<float>(2)[0] = -m_extrinsic.ptr<float>(2)[0];
	flipedExtrinsic.ptr<float>(2)[1] = -m_extrinsic.ptr<float>(2)[1];
	flipedExtrinsic.ptr<float>(2)[2] = -m_extrinsic.ptr<float>(2)[2];
	flipedExtrinsic.ptr<float>(2)[3] = -m_extrinsic.ptr<float>(2)[3];

	return flipedExtrinsic;
}

// get/set R and t (the pose the target object)
void Camera::getRt(cv::Mat &R, cv::Mat &t) const
{
	R = cv::Mat::eye(3, 3, CV_32FC1);
	t = cv::Mat::zeros(3, 1, CV_32FC1);

	// copy R from m_extrinsic
	R.at<float>(0, 0) = m_extrinsic.at<float>(0, 0);
	R.at<float>(0, 1) = m_extrinsic.at<float>(0, 1);
	R.at<float>(0, 2) = m_extrinsic.at<float>(0, 2);
	R.at<float>(1, 0) = m_extrinsic.at<float>(1, 0);
	R.at<float>(1, 1) = m_extrinsic.at<float>(1, 1);
	R.at<float>(1, 2) = m_extrinsic.at<float>(1, 2);
	R.at<float>(2, 0) = m_extrinsic.at<float>(2, 0);
	R.at<float>(2, 1) = m_extrinsic.at<float>(2, 1);
	R.at<float>(2, 2) = m_extrinsic.at<float>(2, 2);

	// copy t from m_extrinsic
	t.at<float>(0) = m_extrinsic.at<float>(0, 3);
	t.at<float>(1) = m_extrinsic.at<float>(1, 3);
	t.at<float>(2) = m_extrinsic.at<float>(2, 3);
}

void Camera::setRt(const cv::Mat &R, const cv::Mat &t)
{
	m_extrinsic.at<float>(0, 0) = R.at<float>(0, 0);
	m_extrinsic.at<float>(0, 1) = R.at<float>(0, 1);
	m_extrinsic.at<float>(0, 2) = R.at<float>(0, 2);
	m_extrinsic.at<float>(1, 0) = R.at<float>(1, 0);
	m_extrinsic.at<float>(1, 1) = R.at<float>(1, 1);
	m_extrinsic.at<float>(1, 2) = R.at<float>(1, 2);
	m_extrinsic.at<float>(2, 0) = R.at<float>(2, 0);
	m_extrinsic.at<float>(2, 1) = R.at<float>(2, 1);
	m_extrinsic.at<float>(2, 2) = R.at<float>(2, 2);
	m_extrinsic.at<float>(0, 3) = t.at<float>(0);
	m_extrinsic.at<float>(1, 3) = t.at<float>(1);
	m_extrinsic.at<float>(2, 3) = t.at<float>(2);
}

// Euler rotation in XZY format to rotation matrix
cv::Mat eulerXZY2Rot(float x, float y, float z)
{
	cv::Mat rotationMatrix = cv::Mat::eye(3,3,CV_32FC1);

	// Assuming the angles are in radians.
	float cx = cos(x);
	float sx = sin(x);
	float cy = cos(y);
	float sy = sin(y);
	float cz = cos(z);
	float sz = sin(z);

	float m00, m01, m02, m10, m11, m12, m20, m21, m22;

	m00 = cy*cz;
	m01 = sx*sy - cx*cy*sz;
	m02 = cx*sy + cy*sx*sz;
	m10 = sz;
	m11 = cx*cz;
	m12 = -cz*sx;
	m20 = -cz*sy;
	m21 = cy*sx + cx*sy*sz;
	m22 = cx*cy - sx*sy*sz;

	rotationMatrix.at<float>(0, 0) = m00;
	rotationMatrix.at<float>(0, 1) = m01;
	rotationMatrix.at<float>(0, 2) = m02;
	rotationMatrix.at<float>(1, 0) = m10;
	rotationMatrix.at<float>(1, 1) = m11;
	rotationMatrix.at<float>(1, 2) = m12;
	rotationMatrix.at<float>(2, 0) = m20;
	rotationMatrix.at<float>(2, 1) = m21;
	rotationMatrix.at<float>(2, 2) = m22;

	return rotationMatrix;
}

// Euler angle in XZY format (rotate around the X axis first)
void Camera::setRt(float ex, float ey, float ez, float tx, float ty, float tz)
{
	cv::Mat R = eulerXZY2Rot(ex, ey, ez);
	
	m_extrinsic.at<float>(0, 0) = R.at<float>(0, 0);
	m_extrinsic.at<float>(0, 1) = R.at<float>(0, 1);
	m_extrinsic.at<float>(0, 2) = R.at<float>(0, 2);
	m_extrinsic.at<float>(1, 0) = R.at<float>(1, 0);
	m_extrinsic.at<float>(1, 1) = R.at<float>(1, 1);
	m_extrinsic.at<float>(1, 2) = R.at<float>(1, 2);
	m_extrinsic.at<float>(2, 0) = R.at<float>(2, 0);
	m_extrinsic.at<float>(2, 1) = R.at<float>(2, 1);
	m_extrinsic.at<float>(2, 2) = R.at<float>(2, 2);
	m_extrinsic.at<float>(0, 3) = tx;
	m_extrinsic.at<float>(1, 3) = ty;
	m_extrinsic.at<float>(2, 3) = tz;
}