// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>
// #include <math.h>
// 
// #include "Traker/Particle.h"
// #include "Traker/Traker.h"
// #include "GLRenderer/include/glRenderer.h"
// #include "POSEST/include/sam.h"
// #include "Traker/Config.h"
// using namespace OD;
// namespace OD{
// int particleCmp(const void* p1, const void* p2)
// {
// 	Particle *p1_ = (Particle*)p1;
// 	Particle *p2_ = (Particle*)p2;
// 
// 	if (p1_->w > p2_->w)
// 		return -1;
// 	if (p1_->w < p2_->w)
// 		return 1;
// 	return 0;
// }
// 
// ParticleFilter::ParticleFilter(int numParticle, float arParam, bool useAR, float noiseRateLow, float noiseRateHigh)
// 	:m_numParticle(numParticle), m_arParam(arParam), m_useAR(useAR), m_noiseRateLow(noiseRateLow), m_noiseRateHigh(noiseRateHigh)
// {
// 	m_particles = (Particle *)malloc(m_numParticle*sizeof(Particle));
// 	particlesTmp = (Particle *)malloc(m_numParticle*sizeof(Particle));
// 
// 	rngstate = (struct rng_state *)malloc(sizeof(struct rng_state));
// 	rng_init(RNG_NONREPEATABLE, rngstate);
// }
// 
// ParticleFilter::~ParticleFilter()
// {
// 	if (m_particles)
// 	{
// 		free(m_particles);
// 		m_particles = NULL;
// 	}
// 	if (particlesTmp)
// 	{
// 		free(particlesTmp);
// 		particlesTmp = NULL;
// 	}
// 	if (rngstate)
// 	{
// 		free(rngstate);
// 		rngstate = NULL;
// 	}
// }
// 
// /**
// Creates an initial distribution of particles by sampling from a Gaussian
// distribution around the specific se(3)
// */
// static void init_(Particle *p, float rt[6], float w)
// {
// 	memcpy(p->rt, rt, 6 * sizeof(float));
// 	memset(p->arVel, 0, 6 * sizeof(float));
// 	memcpy(p->trt, rt, 6 * sizeof(float));
// 	memcpy(p->ort, rt, 6 * sizeof(float));
// 	p->w = w;
// }
// void ParticleFilter::init(float rt[6])
// {
// 	register int i;
// 	float weight = 1.0/m_numParticle;
// #pragma omp parallel for
// 	for (i = 0; i < m_numParticle; ++i)
// 	{
// 		init_(m_particles + i, rt, weight);
// 	}
// }
// 
// /**
// Transit particles by autoregressive dynamics, then add a Gaussian noise
// */
// //#define CALC_IN_SE3
// static void transit_(int i, int numParticles, Particle *p, float arParam, bool useAR, 
//                      struct rng_state* rngstate, float lowNoiseRate, float highNoiseRate)
// {
// 	register int k;
// 	register float *rt = p->rt;
// 	register float *trt = p->trt;
// 	register float *ort = p->ort;
// 	register float *arVel = p->arVel;
// 
// #if CALC_IN_SE3
// 	// random Gaussian noise
// 	const float std_rot = 0.01;     // Need to tune, 0.1radian = 5.7degree
// 	const float std_tra = 0.01;    // Need to tune
// 	float rt_noise_[6];
// 	for (k = 0; k < 3; ++k) rt_noise_[k] = std_tra * rng_stdnormal(rngstate); //for translation
// 	for (k = 3; k < 6; ++k) rt_noise_[k] = std_rot * rng_stdnormal(rngstate); //for rotation
// 	// calculation in SE3
// 	float rt_[6] = { rt[3], rt[4], rt[5], rt[0], rt[1], rt[2] }; // translation vector ,then rotation vector
// 	float arVel_[6] = { arVel[3], arVel[4], arVel[5], arVel[0], arVel[1], arVel[2] };
// 	float trt_[6];
// 	liegroups::SE3<float> rt_SE3, arVel_SE3, trt_SE3, trt__SE3, rt_noise_SE3;
// #else
// 	// random Gaussian noise
// 	const float std_rot = OD::Config::configInstance().PARTICLE_ROTATION_GAUSSIAN_STD;     // Need to tune, 0.1radian = 5.7degree
// 	const float std_tra = OD::Config::configInstance().PARTICLE_TRANSLATION_GAUSSIAN_STD;    // Need to tune
// 	float rt_noise[6];
// 	for (k = 0; k < 3; ++k) rt_noise[k] = std_rot * rng_stdnormal(rngstate); //for rotation 
// 	for (k = 3; k < 6; ++k) rt_noise[k] = std_tra * rng_stdnormal(rngstate); //for translation
// #endif
// 	
// 	if (useAR)
// 	{ 
// 		// use autoregressive dynamics
// #if CALC_IN_SE3
// 		for (k = 0; k < 6; ++k)
// 		{
// 			arVel_[k] *= arParam;	
// 		}
// 
// 		liegroups::exp(rt_SE3, rt_);
// 		liegroups::exp(arVel_SE3, arVel_);
// 		liegroups::multiply(trt_SE3, rt_SE3, arVel_SE3);
// 		liegroups::log(trt_, trt_SE3);
// 
// 		sam_rvecnorm(&trt_[3]);// normlize the rotation vector to [-PI, PI]
// #else
// 		for (k = 0; k < 6; ++k)
// 		{
// 			trt[k] = rt[k] + arParam*arVel[k];
// 		}
// 		sam_rvecnorm(trt);// normlize the rotation vector to [-PI, PI]
// #endif
// 	}
// 	else
// 	{
// #if CALC_IN_SE3
// 		for (k = 0; k < 6; ++k)
// 		{
// 			trt_[k] = rt_[k];
// 		}
// #else
// 		for (k = 0; k < 6; ++k)
// 		{
// 			trt[k] = rt[k];
// 		}
// #endif
// 	}
// 
// 	if (numParticles>=1 && i<numParticles/2)
// 	{
// 		// good particles with low noise rate
// #if CALC_IN_SE3
// 		for (k = 0; k < 6; ++k)
// 		{
// 			rt_noise_[k] *= lowNoiseRate;
// 		}
// #else
// 		for (k = 0; k < 6; ++k)
// 		{
// 			trt[k] += rt_noise[k] * lowNoiseRate;
// 		}
// #endif
// 	}
// 	else if (numParticles>=1 && i>=numParticles/2)
// 	{
// 		// suboptimal particles with high noise rate
// #if CALC_IN_SE3
// 		for (k = 0; k < 6; ++k)
// 		{
// 			rt_noise_[k]  *= highNoiseRate;
// 		}
// #else
// 		for (k = 0; k < 6; ++k)
// 		{
// 			trt[k] += rt_noise[k] * highNoiseRate;
// 		}
// #endif
// 	}
// 
// #if CALC_IN_SE3
// 	liegroups::exp(trt_SE3, trt_);
// 	liegroups::exp(rt_noise_SE3, rt_noise_);
// 	liegroups::multiply(trt__SE3, trt_SE3, rt_noise_SE3);
// 	liegroups::log(trt_, trt__SE3);
// 
// 	// copy to trt
// 	trt[0] = trt_[3]; trt[1] = trt_[4]; trt[2] = trt_[5];
// 	trt[3] = trt_[0]; trt[4] = trt_[1]; trt[5] = trt_[2];
// #else
// #endif
// 
// 	// normlize the rotation vector to [-PI, PI]
// 	sam_rvecnorm(trt);
// 
// 	// copy to ort, the ort is the start point for tracking refinement
// 	memcpy(ort, trt, 6*sizeof(float));
// }
// void ParticleFilter::transit()
// {
// 	register int i;
// #pragma omp parallel for
// 	for (i = 0; i < m_numParticle; ++i)
// 	{
// 		transit_(i, m_numParticle, m_particles + i, m_arParam, m_useAR,
// 			rngstate, m_noiseRateLow, m_noiseRateHigh);
// 	}
// }
// 
// /**
// Update particles by some method,
// for 3d tracking, refine the states by 3d pose tracker
// */
// static float max_e = 1000;
// static void update_(Particle *p, Traker *tracker, cv::Mat frame, cv::Mat prevFrame, float K[9],
// 	int NLrefine, GLRenderer &renderer, bool useAR)
// {
// 	register float *rt = p->rt;
// 	register float *trt = p->trt;
// 	register float *ort = p->ort;
// 	register float *arVel = p->arVel;
// 
// 	// refine the transited rt by 3d tracking method
// 	float e2 = 1E12;
// 	int ret = tracker->optimizingLM(trt,frame, e2);
// 
// 	// ret==0 is OK, ret==-1 is failure
// #if CALC_IN_SE3
// 	float rt_[6] = { rt[3], rt[4], rt[5], rt[0], rt[1], rt[2] }; // translation vector ,then rotation
// 	float ort_[6] = { ort[3], ort[4], ort[5], ort[0], ort[1], ort[2] }; // translation vector ,then rotation vector
// 	liegroups::SE3<float> arVel_SE3, rt_SE3, ort_SE3;
// #endif
// 	if (ret==0)
// 	{
// 		if (useAR)
// 		{
// #if CALC_IN_SE3
// 			liegroups::exp(rt_SE3, rt_);
// 			liegroups::exp(ort_SE3, ort_);
// 
// 			//calculate the ar velocity
// 			liegroups::multiply_a_binv(arVel_SE3, ort_SE3, rt_SE3);
// 			float arVel_[6] = { 0 };
// 			liegroups::log(arVel_, arVel_SE3);
// 
// 			arVel[0] = arVel_[3]; arVel[1] = arVel_[4]; arVel[2] = arVel_[5];
// 			arVel[3] = arVel_[0]; arVel[4] = arVel_[1]; arVel[5] = arVel_[2];
// #else
// 			for (int j = 0; j < 6; ++j)
// 			{
// 				arVel[j] = ort[j] - rt[j];
// 			}
// #endif
// 		}
// 
// 		// update the rt from ort
// 		memcpy(rt, ort, 6*sizeof(float));
// 		p->w = exp(-e/max_e);
// 	}
// 	else
// 	{
// 		p->w = 0;
// 	}
// }
// void ParticleFilter::update(Traker *tracker, cv::Mat frame, cv::Mat prevFrame, float K[9],
// 	int NLrefine, GLRenderer &renderer)
// {
// 	register int i;
// 	for (i = 0; i < m_numParticle; ++i)
// 	{
// 		update_(m_particles + i, tracker, frame, prevFrame, K, NLrefine, renderer, m_useAR);
// 	}
// }
// 
// static void update2_(Particle *p, Traker *tracker, cv::Mat distMap, cv::Mat frame, cv::Mat prevFrame, float K[9],
// 	int NLrefine, GLRenderer &renderer, bool useAR)
// {
// 	register float *rt = p->rt;
// 	register float *trt = p->trt;
// 	register float *ort = p->ort;
// 	register float *arVel = p->arVel;
// 
// 	// refine the transited rt by 3d tracking method
// 	float e = 1E12;
// 	int ret = tracker->track2(distMap, frame, prevFrame, trt, K, NLrefine, renderer, ort, &e);
// 
// 	// ret==0 is OK, ret==-1 is failure
// #if CALC_IN_SE3
// 	float rt_[6] = { rt[3], rt[4], rt[5], rt[0], rt[1], rt[2] }; // translation vector ,then rotation
// 	float ort_[6] = { ort[3], ort[4], ort[5], ort[0], ort[1], ort[2] }; // translation vector ,then rotation
// 	liegroups::SE3<float> arVel_SE3, rt_SE3, ort_SE3;
// #endif
// 	if (ret == 0)
// 	{
// 		if (useAR)
// 		{
// #if CALC_IN_SE3
// 			liegroups::exp(rt_SE3, rt_);
// 			liegroups::exp(ort_SE3, ort_);
// 
// 			//calculate the ar velocity
// 			liegroups::multiply_a_binv(arVel_SE3, ort_SE3, rt_SE3);
// 			float arVel_[6] = { 0 };
// 			liegroups::log(arVel_, arVel_SE3);
// 
// 			arVel[0] = arVel_[3]; arVel[1] = arVel_[4]; arVel[2] = arVel_[5];
// 			arVel[3] = arVel_[0]; arVel[4] = arVel_[1]; arVel[5] = arVel_[2];
// #else
// 			for (int j = 0; j < 6; ++j)
// 			{
// 				arVel[j] = ort[j] - rt[j];
// 			}
// #endif
// 		}
// 
// 		// update the rt from ort
// 		memcpy(rt, ort, 6 * sizeof(float));
// 		p->w = exp(-e / max_e);
// 	}
// 	else
// 	{
// 		p->w = 0;
// 	}
// }
// 
// #if OUT_INTER_RESULTS
// #include "../GLRenderer/include/timer.h"
// static Timer timer;
// static int frame_id = 0;
// #include<fstream>
// static std::ofstream outTimeFile("pf_update_time.txt");
// static cv::VideoWriter outPFBeforeVideo("pf_before.avi", CV_FOURCC('L', 'A', 'G', 'S'), 30.0, cv::Size(640, 480));
// static cv::VideoWriter outPFAfterVideo("pf_after.avi", CV_FOURCC('L', 'A', 'G', 'S'), 30.0, cv::Size(640, 480));
// static cv::VideoWriter outPFBestBeforeVideo("pf_best_before.avi", CV_FOURCC('L', 'A', 'G', 'S'), 30.0, cv::Size(640, 480));
// static cv::VideoWriter outPFBestAfterVideo("pf_best_after.avi", CV_FOURCC('L', 'A', 'G', 'S'), 30.0, cv::Size(640, 480));
// static cv::VideoWriter outPFBestMixVideo("pf_best_mix.avi", CV_FOURCC('L', 'A', 'G', 'S'), 30.0, cv::Size(640, 480));
// #endif
// 
// void ParticleFilter::update2(Traker *tracker, cv::Mat frame, cv::Mat prevFrame, float K[9], int NLrefine,
// 	GLRenderer &renderer)
// {
// 	// frame edge, distance map
// #if OUT_INTER_RESULTS
// 	timer.start();
// #endif
// 
// #if 0
// 	// frame edge, distance map
// 	Mat frameEdge;
// 	float lowThres = 20, highThres = 60; //50, 100
// 	int frameChannels = frame.channels();
// 	if (frameChannels == 1)
// 	{
// 		cv::Canny(frame, frameEdge, lowThres, highThres);
// 	}
// 	if (frameChannels == 3)
// 	{
// 		std::vector<Mat> bgr;
// 		cv::split(frame, bgr);
// 
// 		cv::Canny(bgr[0], bgr[0], lowThres, highThres);
// 		cv::Canny(bgr[1], bgr[1], lowThres, highThres);
// 		cv::Canny(bgr[2], bgr[2], lowThres, highThres);
// 
// 		Mat mergedImage;
// 		bitwise_or(bgr[0], bgr[1], mergedImage);
// 		bitwise_or(mergedImage, bgr[2], frameEdge);
// 	}
// #else
// 	Mat frameGray;
// 	cvtColor(frame, frameGray, CV_BGR2GRAY);
// 	Mat frameEdge;
// 	float lowThres = 20, highThres = 60; //50, 100
// 	cv::blur(frameGray, frameGray, cv::Size(3, 3));
// 	cv::Canny(frameGray, frameEdge, lowThres, highThres);
// #endif
// 
// #if OUT_INTER_RESULTS
// 	timer.stop();
// 	outTimeFile << "frame#" << frame_id << " "
// 		<< "canny:" << timer.getElapsedTimeInMilliSec() << " ";
// 
// 	timer.start();
// #endif
// 
// 	Mat distMap;
// 	cv::distanceTransform(~frameEdge, distMap, CV_DIST_L2, 3);// the distance to zero pixels
// 
// #if OUT_INTER_RESULTS
// 	timer.stop();
// 	outTimeFile << "dt:" << timer.getElapsedTimeInMilliSec() << " ";
// #endif
// 
// #if OUT_INTER_RESULTS
// 	Mat distMap8u;
// 	cv::normalize(distMap, distMap8u, 0, 255, cv::NORM_MINMAX, CV_8UC1);
// 	Mat distMap8uc3Before, distMap8uc3After, distMap8uc3BestBefore, distMap8uc3BestAfter, distMap8uc3BestMix;
// 	cv::cvtColor(distMap8u, distMap8uc3Before, CV_GRAY2BGR);
// 	distMap8uc3After = distMap8uc3Before.clone();
// 	distMap8uc3BestBefore = distMap8uc3Before.clone();
// 	distMap8uc3BestAfter = distMap8uc3Before.clone();
// 	distMap8uc3BestMix = distMap8uc3Before.clone();
// 
// 	float update_t = 0;
// #endif
// 	
// 	register int i;
// 	for (i = 0; i < m_numParticle; ++i)
// 	{
// #if OUT_INTER_RESULTS
// 		timer.start();
// #endif
// 		update2_(m_particles + i, tracker, distMap, frame, prevFrame, K, NLrefine, renderer, m_useAR);
// 
// #if OUT_INTER_RESULTS
// 		timer.stop();
// 		update_t += timer.getElapsedTimeInMilliSec();
// #endif
// 
// #if OUT_INTER_RESULTS
// 		std::vector<cv::Point2f> contour2dBefore;
// 		std::vector<cv::Point3f> &contour3dPoints = tracker->contourPoints;
// 		cv::Mat rVec_before = cv::Mat::zeros(3,1,CV_32FC1);
// 		rVec_before.at<float>(0, 0) = m_particles[i].trt[0];
// 		rVec_before.at<float>(1, 0) = m_particles[i].trt[1];
// 		rVec_before.at<float>(2, 0) = m_particles[i].trt[2];
// 		cv::Mat tVec_before = cv::Mat::zeros(3, 1, CV_32FC1);
// 		tVec_before.at<float>(0, 0) = m_particles[i].trt[3];
// 		tVec_before.at<float>(1, 0) = m_particles[i].trt[4];
// 		tVec_before.at<float>(2, 0) = m_particles[i].trt[5];
// 		cv::projectPoints(contour3dPoints, rVec_before, tVec_before, 
// 			renderer.camera.getIntrinsic(), renderer.camera.getDistorsions(), contour2dBefore);
// 		for (int j = 0; j < contour2dBefore.size(); ++j)
// 		{
// 			cv::line(distMap8uc3Before, cv::Point(contour2dBefore[j].x, contour2dBefore[j].y), cv::Point(contour2dBefore[j].x, contour2dBefore[j].y),
// 				cv::Scalar(0, 0, 255));
// 		}	
// 
// 		std::vector<cv::Point2f> contour2dAfter;
// 		cv::Mat rVec_after = cv::Mat::zeros(3, 1, CV_32FC1);
// 		rVec_after.at<float>(0, 0) = m_particles[i].ort[0];
// 		rVec_after.at<float>(1, 0) = m_particles[i].ort[1];
// 		rVec_after.at<float>(2, 0) = m_particles[i].ort[2];
// 		cv::Mat tVec_after = cv::Mat::zeros(3, 1, CV_32FC1);
// 		tVec_after.at<float>(0, 0) = m_particles[i].ort[3];
// 		tVec_after.at<float>(1, 0) = m_particles[i].ort[4];
// 		tVec_after.at<float>(2, 0) = m_particles[i].ort[5];
// 		cv::projectPoints(contour3dPoints, rVec_after, tVec_after,
// 			renderer.camera.getIntrinsic(), renderer.camera.getDistorsions(), contour2dAfter);
// 		for (int j = 0; j < contour2dAfter.size(); ++j)
// 		{
// 			cv::line(distMap8uc3After, cv::Point(contour2dAfter[j].x, contour2dAfter[j].y), cv::Point(contour2dAfter[j].x, contour2dAfter[j].y),
// 				cv::Scalar(0, 255, 0));
// 		}	
// #endif
// 	}	
// 
// #if OUT_INTER_RESULTS
// 	memcpy(particlesTmp, m_particles, m_numParticle*sizeof(Particle));
// 	qsort(particlesTmp, m_numParticle, sizeof(Particle), particleCmp);
// 	renderer.camera.setExtrinsic(particlesTmp[0].trt[0], particlesTmp[0].trt[1], particlesTmp[0].trt[2],
// 		particlesTmp[0].trt[3], particlesTmp[0].trt[4], particlesTmp[0].trt[5]);
// 	renderer.drawMode = 0;
// 	renderer.bgImgUsed = false;
// 	renderer.render();
// 	Mat fgMask;
// 	cv::normalize(renderer.depthMap, fgMask, 0, 255, cv::NORM_MINMAX, CV_8UC1);
// 	cv::threshold(fgMask, fgMask, 254, 255, CV_THRESH_BINARY_INV); // make sure foreground is white
// 	tracker->sampleContourPoints(fgMask, renderer);
// 
// 	std::vector<cv::Point2f> contour2dAfter, contour2dBefore;
// 	std::vector<cv::Point3f> &contour3dPoints = tracker->contourPoints;
// 	cv::Mat rVec_before = cv::Mat::zeros(3, 1, CV_32FC1);
// 	rVec_before.at<float>(0, 0) = particlesTmp[0].trt[0];
// 	rVec_before.at<float>(1, 0) = particlesTmp[0].trt[1];
// 	rVec_before.at<float>(2, 0) = particlesTmp[0].trt[2];
// 	cv::Mat tVec_before = cv::Mat::zeros(3, 1, CV_32FC1);
// 	tVec_before.at<float>(0, 0) = particlesTmp[0].trt[3];
// 	tVec_before.at<float>(1, 0) = particlesTmp[0].trt[4];
// 	tVec_before.at<float>(2, 0) = particlesTmp[0].trt[5];
// 	cv::projectPoints(contour3dPoints, rVec_before, tVec_before,
// 		renderer.camera.getIntrinsic(), renderer.camera.getDistorsions(), contour2dBefore);
// 	for (int j = 0; j < contour2dBefore.size(); ++j)
// 	{
// 		cv::line(distMap8uc3BestBefore, cv::Point(contour2dBefore[j].x, contour2dBefore[j].y), cv::Point(contour2dBefore[j].x, contour2dBefore[j].y),
// 			cv::Scalar(0, 0, 255));
// 		cv::line(distMap8uc3BestMix, cv::Point(contour2dBefore[j].x, contour2dBefore[j].y), cv::Point(contour2dBefore[j].x, contour2dBefore[j].y),
// 			cv::Scalar(0, 0, 255), 2);
// 	}
// 
// 	cv::Mat rVec_after = cv::Mat::zeros(3, 1, CV_32FC1);
// 	rVec_after.at<float>(0, 0) = particlesTmp[0].ort[0];
// 	rVec_after.at<float>(1, 0) = particlesTmp[0].ort[1];
// 	rVec_after.at<float>(2, 0) = particlesTmp[0].ort[2];
// 	cv::Mat tVec_after = cv::Mat::zeros(3, 1, CV_32FC1);
// 	tVec_after.at<float>(0, 0) = particlesTmp[0].ort[3];
// 	tVec_after.at<float>(1, 0) = particlesTmp[0].ort[4];
// 	tVec_after.at<float>(2, 0) = particlesTmp[0].ort[5];
// 	cv::projectPoints(contour3dPoints, rVec_after, tVec_after,
// 		renderer.camera.getIntrinsic(), renderer.camera.getDistorsions(), contour2dAfter);
// 	for (int j = 0; j < contour2dAfter.size(); ++j)
// 	{
// 		cv::line(distMap8uc3BestAfter, cv::Point(contour2dAfter[j].x, contour2dAfter[j].y), cv::Point(contour2dAfter[j].x, contour2dAfter[j].y),
// 			cv::Scalar(0, 255, 0));
// 		cv::line(distMap8uc3BestMix, cv::Point(contour2dAfter[j].x, contour2dAfter[j].y), cv::Point(contour2dAfter[j].x, contour2dAfter[j].y),
// 			cv::Scalar(0, 255, 0), 2);
// 	}
// 
// 	outPFBeforeVideo.write(distMap8uc3Before);
// 	outPFAfterVideo.write(distMap8uc3After);
// 	outPFBestBeforeVideo.write(distMap8uc3BestBefore);
// 	outPFBestAfterVideo.write(distMap8uc3BestAfter);
// 	outPFBestMixVideo.write(distMap8uc3BestMix);
// 
// 	cv::imwrite("distMap8uc3Before.png", distMap8uc3Before);
// 	cv::imwrite("distMap8uc3After.png", distMap8uc3After);
// 	cv::imwrite("distMap8uc3BestBefore.png", distMap8uc3BestBefore);
// 	cv::imwrite("distMap8uc3BestAfter.png", distMap8uc3BestAfter);
// 	cv::imwrite("distMap8uc3BestMix.png", distMap8uc3BestMix);
// 
// 	outTimeFile << "update_:" << update_t << "\n";
// 
// 	frame_id++;
// #endif
// }
// 
// /**
// Normalizes particle weights so they sum to 1
// */
// void ParticleFilter::normalizeWeights()
// {
// 	register int i;
// 	register float sum=0;
// #pragma omp parallel for reduction(+:sum)
// 	for (i = 0; i < m_numParticle; ++i)
// 	{
// 		sum += m_particles[i].w;
// 	}
// 
// 	if (sum>1e-12)
// 	{
// #pragma omp parallel for
// 		for (i = 0; i < m_numParticle; ++i)
// 		{
// 			m_particles[i].w /= sum;
// 		}
// 	}
// }
// 
// /**
// Re-samples a set of weighted particles to produce a new set of unweighted
// particles
// Classic implementation is Roulette algorithm, here we use a simple method
// */
// void ParticleFilter::resample()
// {
// 	normalizeWeights();
// 	qsort(m_particles, m_numParticle, sizeof(Particle), particleCmp);
// 
// 	register int i, j, k=0;
// 	for (i = 0; i < m_numParticle; ++i)
// 	{
// 		int np = static_cast<int>( round(m_particles[i].w * m_numParticle) );
// 		for (j = 0; j < np; ++j)
// 		{
// 			particlesTmp[k++] = m_particles[i];
// 			if (k == m_numParticle)
// 			{
// 				Particle *forSwappingParticles = m_particles;
// 				m_particles = particlesTmp;
// 				particlesTmp = forSwappingParticles;
// 				return;
// 			}
// 		}
// 	}
// 	while (k<m_numParticle)
// 	{
// 		particlesTmp[k++] = m_particles[0];
// 	}
// 
// 	// swap the memory pointer between m_particles and particlesTmp 
// 	Particle *forSwappingParticles = m_particles;
// 	m_particles = particlesTmp;
// 	particlesTmp = forSwappingParticles;
// }
// }