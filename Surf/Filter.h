#ifndef _FILTER_H_
#define _FILTER_H_

#include "cv.h"

#define icvSetZero_32f( dst, cols, rows ) memset((dst),0,(rows)*(cols)*sizeof(float))
#define icvCheckVector_32f( ptr, len )
#define icvTransformVector_32f( matr, src, dst, w, h ) \
    icvMulMatrix_32f( matr, w, h, src, 1, w, dst )
#define icvCopyVector_32f( src, len, dst ) memcpy((dst),(src),(len)*sizeof(float))
typedef struct CvParticleFilter
{
    int MP;                 /* Dimension of the measurement vector         */
    int DP;                 /* Dimension of the state vector   */
    float* DynamMatr;       /* Matrix of the linear Dynamics system */
    float* State;           /* Vector of State                       */
    int SamplesNum;         /* Number of the Samples                 */
    float** flSamples;      /* array of the Sample Vectors           */
    float** flNewSamples;   /* temporary array of the Sample Vectors */
    float* flConfidence;    /* Confidence for each Sample            */
    float* flCumulative;    /* Cumulative confidence                 */
    float* Temp;            /* Temporary vector                      */
    float* RandomSample;    /* RandomVector to update sample set     */
    struct CvRandState* RandS; /* Array of structures to generate random vectors */
}
CvParticleFilter;


CV_INLINE void icvScaleVector_32f( const float* src, float* dst,
								  int len, double scale );
CV_INLINE void icvAddVector_32f( const float* src1, const float* src2,
								float* dst, int len );
CV_INLINE void icvMulMatrix_32f( const float* src1, int w1, int h1,
								const float* src2, int w2, int h2,
								float* dst );
/* Creates Particle Filter state */
CVAPI(CvParticleFilter*)  cvCreateParticleFilter( int dynam_params,
												 int measure_params,
												 int sample_count );

/* Releases Particle Filter state */
CVAPI(void)  cvReleaseParticleFilter( CvParticleFilter** ParticleFilter );

/* Updates Particle Filter by time (predict future state of the system) */
CVAPI(void)  cvParticleFilterPredict( CvParticleFilter* ParticleFilter);

/* Initializes Particle Filter sample set */
CVAPI(void)  cvParticleFilterInitSampleSet( CvParticleFilter* ParticleFilter, CvMat* lower_bound, CvMat* upper_bound );

/* Propagates Particle Filter sample set */
CVAPI(void)  cvParticleFilterPropagate( CvParticleFilter* ParticleFilter);

/* Resamples Particle Filter sample set */
CVAPI(void)  cvParticleFilterResample( CvParticleFilter* ParticleFilter);

/* Generates expectation of the state */
CVAPI(void)  cvParticleFilterExpectation( CvParticleFilter* ParticleFilter);

#endif