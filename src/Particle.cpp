#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "Traker/Particle.h"
#include "Traker/Traker.h"
#include "GLRenderer/include/glRenderer.h"
#include "POSEST/include/sam.h"
#include "Traker/Config.h"
using namespace OD;
namespace OD{
int particleCmp(const void* p1, const void* p2)
{
	Particle *p1_ = (Particle*)p1;
	Particle *p2_ = (Particle*)p2;

	if (p1_->w > p2_->w)
		return -1;
	if (p1_->w < p2_->w)
		return 1;
	return 0;
}

ParticleFilter::ParticleFilter(int numParticle, float arParam, bool useAR, float noiseRateLow, float noiseRateHigh)
	:m_numParticle(numParticle), m_arParam(arParam), m_useAR(useAR), m_noiseRateLow(noiseRateLow), m_noiseRateHigh(noiseRateHigh)
{
	m_particles = (Particle *)malloc(m_numParticle*sizeof(Particle));
	particlesTmp = (Particle *)malloc(m_numParticle*sizeof(Particle));

	rngstate = (struct rng_state *)malloc(sizeof(struct rng_state));
	rng_init(RNG_NONREPEATABLE, rngstate);
}

ParticleFilter::~ParticleFilter()
{
	if (m_particles)
	{
		free(m_particles);
		m_particles = NULL;
	}
	if (particlesTmp)
	{
		free(particlesTmp);
		particlesTmp = NULL;
	}
	if (rngstate)
	{
		free(rngstate);
		rngstate = NULL;
	}
}

/**
Creates an initial distribution of particles by sampling from a Gaussian
distribution around the specific se(3)
*/
static void init_(Particle *p, float rt[6], float w)
{
	memcpy(p->rt, rt, 6 * sizeof(float));
	memset(p->arVel, 0, 6 * sizeof(float));
	memcpy(p->trt, rt, 6 * sizeof(float));
	memcpy(p->ort, rt, 6 * sizeof(float));
	p->w = w;
}
void ParticleFilter::init(float rt[6])
{
	register int i;
	float weight = 1.0/m_numParticle;
#pragma omp parallel for
	for (i = 0; i < m_numParticle; ++i)
	{
		init_(m_particles + i, rt, weight);
	}
}

/**
Transit particles by autoregressive dynamics, then add a Gaussian noise
*/
//#define CALC_IN_SE3
static void transit_(int i, int numParticles, Particle *p, float arParam, bool useAR, 
                     struct rng_state* rngstate, float lowNoiseRate, float highNoiseRate)
{
	register int k;
	register float *rt = p->rt;
	register float *trt = p->trt;
	register float *ort = p->ort;
	register float *arVel = p->arVel;

#if CALC_IN_SE3
	// random Gaussian noise
	const float std_rot = 0.01;     // Need to tune, 0.1radian = 5.7degree
	const float std_tra = 0.01;    // Need to tune
	float rt_noise_[6];
	for (k = 0; k < 3; ++k) rt_noise_[k] = std_tra * rng_stdnormal(rngstate); //for translation
	for (k = 3; k < 6; ++k) rt_noise_[k] = std_rot * rng_stdnormal(rngstate); //for rotation
	// calculation in SE3
	float rt_[6] = { rt[3], rt[4], rt[5], rt[0], rt[1], rt[2] }; // translation vector ,then rotation vector
	float arVel_[6] = { arVel[3], arVel[4], arVel[5], arVel[0], arVel[1], arVel[2] };
	float trt_[6];
	liegroups::SE3<float> rt_SE3, arVel_SE3, trt_SE3, trt__SE3, rt_noise_SE3;
#else
	// random Gaussian noise
	const float std_rot = OD::Config::configInstance().PARTICLE_ROTATION_GAUSSIAN_STD;     // Need to tune, 0.1radian = 5.7degree
	const float std_tra = OD::Config::configInstance().PARTICLE_TRANSLATION_GAUSSIAN_STD;    // Need to tune
	float rt_noise[6];
	for (k = 0; k < 3; ++k) rt_noise[k] = std_rot * rng_stdnormal(rngstate); //for rotation 
	for (k = 3; k < 6; ++k) rt_noise[k] = std_tra * rng_stdnormal(rngstate); //for translation
#endif
	
	if (useAR)
	{ 
		// use autoregressive dynamics
#if CALC_IN_SE3
		for (k = 0; k < 6; ++k)
		{
			arVel_[k] *= arParam;	
		}

		liegroups::exp(rt_SE3, rt_);
		liegroups::exp(arVel_SE3, arVel_);
		liegroups::multiply(trt_SE3, rt_SE3, arVel_SE3);
		liegroups::log(trt_, trt_SE3);

		sam_rvecnorm(&trt_[3]);// normlize the rotation vector to [-PI, PI]
#else
		for (k = 0; k < 6; ++k)
		{
			trt[k] = rt[k] + arParam*arVel[k];
		}
		sam_rvecnorm((double *)trt);// normlize the rotation vector to [-PI, PI]
#endif
	}
	else
	{
#if CALC_IN_SE3
		for (k = 0; k < 6; ++k)
		{
			trt_[k] = rt_[k];
		}
#else
		for (k = 0; k < 6; ++k)
		{
			trt[k] = rt[k];
		}
#endif
	}

	if (numParticles>=1 && i<numParticles/2)
	{
		// good particles with low noise rate
#if CALC_IN_SE3
		for (k = 0; k < 6; ++k)
		{
			rt_noise_[k] *= lowNoiseRate;
		}
#else
		for (k = 0; k < 6; ++k)
		{
			trt[k] += rt_noise[k] * lowNoiseRate;
		}
#endif
	}
	else if (numParticles>=1 && i>=numParticles/2)
	{
		// suboptimal particles with high noise rate
#if CALC_IN_SE3
		for (k = 0; k < 6; ++k)
		{
			rt_noise_[k]  *= highNoiseRate;
		}
#else
		for (k = 0; k < 6; ++k)
		{
			trt[k] += rt_noise[k] * highNoiseRate;
		}
#endif
	}

#if CALC_IN_SE3
	liegroups::exp(trt_SE3, trt_);
	liegroups::exp(rt_noise_SE3, rt_noise_);
	liegroups::multiply(trt__SE3, trt_SE3, rt_noise_SE3);
	liegroups::log(trt_, trt__SE3);

	// copy to trt
	trt[0] = trt_[3]; trt[1] = trt_[4]; trt[2] = trt_[5];
	trt[3] = trt_[0]; trt[4] = trt_[1]; trt[5] = trt_[2];
#else
#endif

	// normlize the rotation vector to [-PI, PI]
	sam_rvecnorm((double *)trt);

	// copy to ort, the ort is the start point for tracking refinement
	memcpy(ort, trt, 6*sizeof(float));
}
void ParticleFilter::transit()
{
	register int i;
#pragma omp parallel for
	for (i = 0; i < m_numParticle; ++i)
	{
		transit_(i, m_numParticle, m_particles + i, m_arParam, m_useAR,
			rngstate, m_noiseRateLow, m_noiseRateHigh);
	}
}

/**
Update particles by some method,
for 3d tracking, refine the states by 3d pose tracker
*/
static float max_e = 1000;
static void update_(Particle *p, Traker *tracker, cv::Mat frame, cv::Mat prevFrame, float K[9],
	int frame_id, GLRenderer &renderer, bool useAR)
{
	register float *rt = p->rt;
	register float *trt = p->trt;
	register float *ort = p->ort;
	register float *arVel = p->arVel;

	// refine the transited rt by 3d tracking method
	float e2 = 1E12;
	int ret = tracker->toTrack(trt,frame,frame_id,ort, e2);

	// ret==0 is OK, ret==-1 is failure
#if CALC_IN_SE3
	float rt_[6] = { rt[3], rt[4], rt[5], rt[0], rt[1], rt[2] }; // translation vector ,then rotation
	float ort_[6] = { ort[3], ort[4], ort[5], ort[0], ort[1], ort[2] }; // translation vector ,then rotation vector
	liegroups::SE3<float> arVel_SE3, rt_SE3, ort_SE3;
#endif
	if (ret==0)
	{
		if (useAR)
		{
#if CALC_IN_SE3
			liegroups::exp(rt_SE3, rt_);
			liegroups::exp(ort_SE3, ort_);

			//calculate the ar velocity
			liegroups::multiply_a_binv(arVel_SE3, ort_SE3, rt_SE3);
			float arVel_[6] = { 0 };
			liegroups::log(arVel_, arVel_SE3);

			arVel[0] = arVel_[3]; arVel[1] = arVel_[4]; arVel[2] = arVel_[5];
			arVel[3] = arVel_[0]; arVel[4] = arVel_[1]; arVel[5] = arVel_[2];
#else
			for (int j = 0; j < 6; ++j)
			{
				arVel[j] = ort[j] - rt[j];
			}
#endif
		}

		// update the rt from ort
		memcpy(rt, ort, 6*sizeof(float));
		p->w = exp(-e2/max_e);
	  
	}
	else
	{
		p->w = 0;
	}
}
void ParticleFilter::update(Traker *tracker, cv::Mat frame, cv::Mat prevFrame, float K[9],
	int frame_id, GLRenderer &renderer)
{
	register int i;
	for (i = 0; i < m_numParticle; ++i)
	{
		update_(m_particles + i, tracker, frame, prevFrame, K, frame_id, renderer, m_useAR);
	}
}


#if OUT_INTER_RESULTS
#include "../GLRenderer/include/timer.h"
static Timer timer;
static int frame_id = 0;
#include<fstream>
static std::ofstream outTimeFile("pf_update_time.txt");
static cv::VideoWriter outPFBeforeVideo("pf_before.avi", CV_FOURCC('L', 'A', 'G', 'S'), 30.0, cv::Size(640, 480));
static cv::VideoWriter outPFAfterVideo("pf_after.avi", CV_FOURCC('L', 'A', 'G', 'S'), 30.0, cv::Size(640, 480));
static cv::VideoWriter outPFBestBeforeVideo("pf_best_before.avi", CV_FOURCC('L', 'A', 'G', 'S'), 30.0, cv::Size(640, 480));
static cv::VideoWriter outPFBestAfterVideo("pf_best_after.avi", CV_FOURCC('L', 'A', 'G', 'S'), 30.0, cv::Size(640, 480));
static cv::VideoWriter outPFBestMixVideo("pf_best_mix.avi", CV_FOURCC('L', 'A', 'G', 'S'), 30.0, cv::Size(640, 480));
#endif



/**
Normalizes particle weights so they sum to 1
*/
void ParticleFilter::normalizeWeights()
{
	register int i;
	register float sum=0;
#pragma omp parallel for reduction(+:sum)
	for (i = 0; i < m_numParticle; ++i)
	{
		sum += m_particles[i].w;
	}

	if (sum>1e-12)
	{
#pragma omp parallel for
		for (i = 0; i < m_numParticle; ++i)
		{
			m_particles[i].w /= sum;
		}
	}
}

/**
Re-samples a set of weighted particles to produce a new set of unweighted
particles
Classic implementation is Roulette algorithm, here we use a simple method
*/
void ParticleFilter::resample()
{
	normalizeWeights();
	qsort(m_particles, m_numParticle, sizeof(Particle), particleCmp);

	register int i, j, k=0;
	for (i = 0; i < m_numParticle; ++i)
	{
		int np = static_cast<int>( round(m_particles[i].w * m_numParticle) );
		for (j = 0; j < np; ++j)
		{
			particlesTmp[k++] = m_particles[i];
			if (k == m_numParticle)
			{
				Particle *forSwappingParticles = m_particles;
				m_particles = particlesTmp;
				particlesTmp = forSwappingParticles;
				return;
			}
		}
	}
	while (k<m_numParticle)
	{
		particlesTmp[k++] = m_particles[0];
	}

	// swap the memory pointer between m_particles and particlesTmp 
	Particle *forSwappingParticles = m_particles;
	m_particles = particlesTmp;
	particlesTmp = forSwappingParticles;
}
}