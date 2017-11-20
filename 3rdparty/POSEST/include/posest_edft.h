/*
//////////////////////////////////////////////////////////////////////////////////////////////////////////
// 
//  Non-linear calibrated camera pose estimation from contour matching in edge distance field
//  the functions are called by EdgeDistanceFieldTracker
//  Bin Wang (binwangsdu@gmail.com)
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////
*/

#ifndef _POSEST_EDFT_H_
#define _POSEST_EDFT_H_

/* define the following if you want to build a DLL with MSVC */
/**
#define DLL_BUILD 
**/

#ifdef __cplusplus
extern "C" {
#endif

	/* non-linear refinement cost functions */
#define POSEST_EDFT_ERR_NLN        0 /* contour matching error with non-linear minimization */
#define POSEST_EDFT_ERR_NLN_MLSL   1 /* contour matching error with non-linear minimization & multistart scheme */

#define NUM_RTPARAMS      6 /* #params involved in rotation + translation (pose) */
#define NUM_RTFPARAMS     7 /* #params involved in rotation + translation+ focal length */
#define NUM_PPARAMS       12 /* #params involved in projection matrix */

/* use as: extern POSEST_API_MOD int POSEST_CALL_CONV func(...) */
#if defined(DLL_BUILD) && defined(_MSC_VER) /* build DLLs with MSVC only! */
#define POSEST_API_MOD    __declspec(dllexport)
#define POSEST_CALL_CONV  __cdecl
#else /* define empty */
#define POSEST_API_MOD 
#define POSEST_CALL_CONV
#endif /* DLL_BUILD && _MSC_VER */

#define POSEST_ERR     -1
#define POSEST_OK       0


/* posest_edft.c */
typedef unsigned char uchar;
/*
** distMap: distance map
** ctrPts3D: 3d points on the contour
** nCtrPts: the number of 3d points on the contour
** width: the image width of distMap and imgIntensity
** height: the image height of distMap and imgIntensity
** K: the pre-calibrated camera intrinsic matrix
** pp: the pose parameters that will be estimated
** npp: the number of pose parameters
** NLRefine: POSEST_EDFT_ERR_NLN or POSEST_EDFT_ERR_NLN_MLSL
** verbose: the flag for controlling information output (verbose>0: outputing information)
** final_e: the residual after optimization
*/
extern int posest_edft(float *distMap, double(*ctrPts3D)[3], int nCtrPts, int width, int height,
	double K[9], double *pp, int npp, int NLRefine, int verbose, double *final_e);

extern void posest_edft_PfromKRt(double P[NUM_PPARAMS], double K[9], double rt[NUM_RTPARAMS]);


#ifdef __cplusplus
}
#endif

#endif /* _POSEST_EDFT_H_ */
