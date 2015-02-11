#ifndef SURF_INLUDE_H
#define SURF_INLUDE_H
#include <cv.h>

#define CV_SURF_EXTENDED (1)


#endif  

typedef struct CvSURFDescriptor
{
	int x;
	int y;
	int laplacian;
	double s;
	double dir;
	double mod;
	double vector[128];
} CvSURFDescriptor;

typedef struct CvSURFPointOne
{
	int x;
	int y;
	int laplacian;
	int size;
	int octave;
	int scale;
} CvSURFPointOne;
 
CvSeq* cvSURFDescriptor( const CvArr* _img, CvMemStorage* storage, double quality, int flags = 0 );

void cvSURFInitialize();

CvSeq* cvSURFFindPair( CvSeq* ImageDescriptor, CvSeq* ObjectDescriptor, CvMemStorage* storage, int flags );


