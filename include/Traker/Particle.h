#ifndef _PARTICLES_H_
#define _PARTICLES_H_

#include <opencv2/opencv.hpp>

#include "POSEST/include/rngs.h"
namespace OD{
class Tracker;
class GLRenderer;

/**
A particle is an instantiation of the state variables of the system
being monitored.  A collection of particles is essentially a
discretization of the posterior probability of the system.
*/
typedef struct Particle 
{
	/*
	(rx, ry, rz, tx, ty, tz) is current state, for 3d tracking it is in se(3)
	*/
	float rt[6];

	/*
	autoregressive velocity for each particle
	*/ 
	float arVel[6];

	/*
	transited rt
	*/
	float trt[6];

	/*
	optimized rt
	*/
	float ort[6];

	/* 
	weight is the importance of the particle
	*/
	float w;   
} Particle;

/*
Compare two particles based on weight.  For use in qsort.
Returns -1 if the p1 has lower weight than p2, 
1 if p1 has higher weight than p2, and 0 if their weights are equal.
*/
int particleCmp(const void* p1, const void* p2);


/**
A particle filter is responsible for updating the state of particles.
*/

class ParticleFilter
{
public:
	ParticleFilter(int numParticle = 1, float arParam = 0.01, bool useAR = false, float noiseRateLow = 1.0, float noiseRateHigh = 1.0);
	~ParticleFilter();

	/**
	Creates an initial distribution of particles by sampling from a Gaussian
	distribution around the specific se(3)
	*/
	void init(float rt[6]);

	/**
	Transit particles by autoregressive dynamic, then add a Gaussian noise
	*/
	void transit();

	/**
	Update particles by some method,
	for 3d tracking, refine the states by 3d pose tracker
	*/
	void update(Tracker *tracker, cv::Mat frame, cv::Mat prevFrame, float K[9], int NLrefine,
		GLRenderer &renderer);

	void update2(Tracker *tracker, cv::Mat frame, cv::Mat prevFrame, float K[9], int NLrefine,
		GLRenderer &renderer); // this method for reducing repeatable computation in tracking

	/**
	Normalizes particle weights so they sum to 1
	*/
	void normalizeWeights();

	/**
	Re-samples a set of weighted particles to produce a new set of unweighted
	particles
	*/
	void resample();

public:
	int m_numParticle;        /* the number of particles */
	Particle *m_particles;    /* the particles */
	bool m_useAR;             /* the flag of using autoregressive*/
	float m_arParam;         /* the coefficient of autoregressive dynamics */
	float m_noiseRateLow;    /* the coefficient of noise for good particles */
	float m_noiseRateHigh;   /* the coefficient of noise for suboptimal particles */

private:
	struct rng_state *rngstate; /* the random number generator*/
	Particle *particlesTmp;    /* the temporary particles for swapping states when resampling*/
};
}
#endif // !_PARTICLES_H_