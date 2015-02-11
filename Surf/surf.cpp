#include "surf.h"
//#include <iostream>
#include <stdio.h>

#define ScanOctave (3)
#define FilterScale (4)
#define SamplingStep (1)

CV_INLINE CvSURFPointOne cvSURFPoint( int x, int y, int laplacian, int size, int octave, int scale )
{
    CvSURFPointOne p;
    p.x = x;
    p.y = y;
    p.laplacian = laplacian;
    p.size = size;
    p.octave = octave;
    p.scale = scale;
    return p;
}

CV_INLINE double
icvCalHaarPattern( int* origin,
		   int* t,
		   int widthStep )
{
	double d = 0;
	int *p0 = 0, *p1 = 0, *p2 = 0, *p3 = 0;
	int n = t[0];
	for ( int k = 0; k < n; k++ )
	{
		p0 = origin+t[1]+t[2]*widthStep;
		p1 = origin+t[1]+t[4]*widthStep;
		p2 = origin+t[3]+t[2]*widthStep;
		p3 = origin+t[3]+t[4]*widthStep;
		d += (double)((*p3-*p2-*p1+*p0)*t[6])/(double)(t[5]);
		t+=6;
	}
	return d;
}

CV_INLINE void
icvResizeHaarPattern( int* t_s,
		      int* t_d,
		      int OldSize,
		      int NewSize )
{
	int n = t_d[0] = t_s[0];
	for ( int k = 0; k < n; k++ )
	{
		t_d[1] = t_s[1]*NewSize/OldSize;
		t_d[2] = t_s[2]*NewSize/OldSize;
		t_d[3] = t_s[3]*NewSize/OldSize;
		t_d[4] = t_s[4]*NewSize/OldSize;
		t_d[5] = (t_d[3]-t_d[1]+1)*(t_d[4]-t_d[2]+1);
		t_d[6] = t_s[6];
		t_d+=6;
		t_s+=6;
	}
}

template<typename Number>
CV_INLINE int
icvSign( Number x )
{
	return (( x < 0 ) ? -1 : 1);
}

CvSeq*
icvFastHessianDetector( const CvMat* sum,
			CvMemStorage* storage,
			double quality )
{
	//double t = (double)cvGetTickCount();
	CvSeq* points = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvSURFPointOne), storage );
	CvMat* hessians[ScanOctave*(FilterScale+2)];
	CvMat* traces[ScanOctave*(FilterScale+2)];
	int size, size_cache[ScanOctave*(FilterScale+2)];
	int scale, scale_cache[ScanOctave*(FilterScale+2)];
	double *hessian_ptr, *hessian_ptr_cache[ScanOctave*(FilterScale+2)];
	double *trace_ptr, *trace_ptr_cache[ScanOctave*(FilterScale+2)];
	int dx_s[] = { 3, 0, 2, 3, 7, 0, 1, 3, 2, 6, 7, 0, -2, 6, 2, 9, 7, 0, 1 };
	int dy_s[] = { 3, 2, 0, 7, 3, 0, 1, 2, 3, 7, 6, 0, -2, 2, 6, 7, 9, 0, 1 };
	int dxy_s[] = { 4, 1, 1, 4, 4, 0, 1, 5, 1, 8, 4, 0, -1, 1, 5, 4, 8, 0, -1, 5, 5, 8, 8, 0, 1 };
	int dx_t[] = { 3, 0, 2, 3, 7, 0, 1, 3, 2, 6, 7, 0, -2, 6, 2, 9, 7, 0, 1 };
	int dy_t[] = { 3, 2, 0, 7, 3, 0, 1, 2, 3, 7, 6, 0, -2, 2, 6, 7, 9, 0, 1 };
	int dxy_t[] = { 4, 1, 1, 4, 4, 0, 1, 5, 1, 8, 4, 0, -1, 1, 5, 4, 8, 0, -1, 5, 5, 8, 8, 0, 1 };
	double dx = 0, dy = 0, dxy = 0;
	int k = 0;
	int hessian_rows, hessian_rows_cache[ScanOctave*(FilterScale+2)];
	int hessian_cols, hessian_cols_cache[ScanOctave*(FilterScale+2)];
	/* hessian detector */
	int o;
	for ( o = 0; o < ScanOctave; o++ )
	{
		//t = (double)cvGetTickCount();
		for ( int s = -1; s < FilterScale+1; s++ )
		{
			if ( s < 0 )
				size_cache[k] = size = 7<<o; // gaussian scale 1.0;
			else
				size_cache[k] = size = (s*6+9)<<o; // gaussian scale size*1.2/9.;
			scale_cache[k] = scale = MAX( size, 9 )*SamplingStep;
			hessian_rows_cache[k] = hessian_rows = (sum->rows)*9/scale;
			hessian_cols_cache[k] = hessian_cols = (sum->cols)*9/scale;
			hessians[k] = cvCreateMat( hessian_rows, hessian_cols, CV_64FC1 );
			traces[k] = cvCreateMat( hessian_rows, hessian_cols, CV_64FC1 );
			int* sum_ptr = (int*)sum->data.ptr;
			icvResizeHaarPattern( dx_s, dx_t, 9, size );
			icvResizeHaarPattern( dy_s, dy_t, 9, size );
			icvResizeHaarPattern( dxy_s, dxy_t, 9, size );
			hessian_ptr_cache[k] = hessian_ptr = (double*)hessians[k]->data.ptr;
			trace_ptr_cache[k] = trace_ptr = (double*)traces[k]->data.ptr;
			hessian_ptr+=4/SamplingStep+(4/SamplingStep)*hessian_cols;
			trace_ptr+=4/SamplingStep+(4/SamplingStep)*hessian_cols;
			int oy = 0, y = 0;
			for ( int j = 0; j < hessian_rows-9/SamplingStep; j++ )
			{
				int * sum_line_ptr = sum_ptr;
				double* trace_line_ptr = trace_ptr;
				double* hessian_line_ptr = hessian_ptr;
				int ox = 0, x = 0;
				for ( int i = 0; i < hessian_cols-9/SamplingStep; i++ )
				{
					dx = icvCalHaarPattern( sum_line_ptr, dx_t, sum->cols );
					dy = icvCalHaarPattern( sum_line_ptr, dy_t, sum->cols );
					dxy = icvCalHaarPattern( sum_line_ptr, dxy_t, sum->cols );
					*hessian_line_ptr = (dx*dy-dxy*dxy*0.81);
					*trace_line_ptr = dx+dy;
					x = (i+1)*scale/9;
					sum_line_ptr+=x-ox;
					ox = x;
					trace_line_ptr++;
					hessian_line_ptr++;
				}
				y = (j+1)*scale/9;
				sum_ptr+=(y-oy)*sum->cols;
				oy = y;
				trace_ptr+=hessian_cols;
				hessian_ptr+=hessian_cols;
			}
			k++;
		}
		//t = (double)cvGetTickCount()-t;
	//	printf( "octave time = %gms\n", t/((double)cvGetTickFrequency()*1000.) );
	}
	double min_accept = quality*300;
	//t = (double)cvGetTickCount()-t;
	//printf( "hessian filter time = %gms\n", t/((double)cvGetTickFrequency()*1000.) );
       // t = (double)cvGetTickCount();
	k = 0;
	for ( o = 0; o < ScanOctave; o++ )
	{
		k++;
		for ( int s = 0; s < FilterScale; s++ )
		{
			size = size_cache[k];
			scale = scale_cache[k];
			hessian_rows = hessian_rows_cache[k];
			hessian_cols = hessian_cols_cache[k];
			int margin = (5/SamplingStep)*scale_cache[k+1]/scale;
			hessian_ptr = hessian_ptr_cache[k]+margin+margin*hessian_cols;
			trace_ptr = trace_ptr_cache[k];
			for ( int j = margin; j < hessian_rows-margin; j++ )
			{
				double* hessian_line_ptr = hessian_ptr;
				for ( int i = margin; i < hessian_cols-margin; i++ )
				{
					if ( *hessian_line_ptr > min_accept )
					{
						bool suppressed = false;
						/* non-maxima suppression */
						for ( int z = k-1; z < k+2; z++ )
						{
							double* temp_hessian_ptr = hessian_ptr_cache[z]+i*scale/scale_cache[z]-1+(j*scale/scale_cache[z]-1)*hessian_cols_cache[z];
							for ( int y = 0; y < 3; y++ )
							{
								double* temp_hessian_line_ptr = temp_hessian_ptr;
								for ( int x = 0; x < 3; x++ )
								{
									if ((( z != k )||( y != 1 )||( x != 1 ))&&( *temp_hessian_line_ptr > *hessian_line_ptr ))
									{
										suppressed = true;
										break;
									}
									temp_hessian_line_ptr++;
								}
								if ( suppressed )
									break;
								temp_hessian_ptr+=hessian_cols_cache[z];
							}
							if ( suppressed )
								break;
						}
						if ( !suppressed )
						{
							CvSURFPointOne point = cvSURFPoint( i*scale/9, j*scale/9, icvSign(trace_ptr[i+j*hessian_cols]), size_cache[k], o, s );
							cvSeqPush( points, &point );
						}
					}
					hessian_line_ptr++;
				}
				hessian_ptr+=hessian_cols;
			}
			k++;
		}
		k++;
	}
	k = 0;
	for (  o = 0; o < ScanOctave; o++ )
		for ( int s = -1; s < FilterScale+1; s++ )
		{
			cvReleaseMat( &hessians[k] );
			cvReleaseMat( &traces[k] );
			k++;
		}
      //  t = (double)cvGetTickCount()-t;
       // printf( "hessian selector time = %gms\n", t/((double)cvGetTickFrequency()*1000.) );
	return points;
}



void
icvSURFGaussian( CvMat* mat, double s )
{
	int w = mat->cols;
	int h = mat->rows;
	double x, y;
	double c2 = 1./(s*s*2);
	double over_exp = 1./(3.14159*2*s*s);
	for ( int i = 0; i < w; i++ )
		for ( int j = 0; j < h; j++ )
		{
			x = i-w/2.;
			y = j-h/2.;
			cvmSet( mat, j, i, exp(-(x*x+y*y)*c2)*over_exp );
		}
}

CvMat* wrap = 0;
IplImage* regions_cache[ScanOctave*FilterScale];
IplImage* region_cache;
CvMat* dx_cache;
CvMat* dy_cache;
CvMat* gauss_kernel_cache;
double CosCache[3600];
double SinCache[3600];

void
cvSURFInitialize()
{
	wrap = cvCreateMat( 2, 3, CV_32FC1 );
	int k = 0;
	for ( int o = 0; o < ScanOctave; o++ )
		for ( int s = 0; s < FilterScale; s++ )
		{
			double scal = ((s*6+9)<<o)*1.2/9.;
			regions_cache[k] = cvCreateImage( cvSize(cvRound(21*scal), cvRound(21*scal)), 8, 1 );
			k++;
		}
		region_cache = cvCreateImage( cvSize(21, 21), 8, 1 );
		dx_cache = cvCreateMat( 20, 20, CV_64FC1 );
		dy_cache = cvCreateMat( 20, 20, CV_64FC1 );
		gauss_kernel_cache = cvCreateMat( 20, 20, CV_64FC1 );
		icvSURFGaussian( gauss_kernel_cache, 3.3 );
		for ( int i = 0; i < 3600; i++ )
		{
			CosCache[i] = cos(i*0.001745329);
			SinCache[i] = sin(i*0.001745329);
		}
}

CvSeq*
cvSURFDescriptor( const CvArr* _img,
		  CvMemStorage* storage,
		  double quality,
		  int flags )
{
	IplImage* img = (IplImage*)_img;
	CvMat* sum = 0;
	sum = cvCreateMat( img->roi->height+1, img->roi->width+1, CV_32SC1 );
	
	cvIntegral( img, sum );
	
	CvMemStorage* point_storage = cvCreateChildMemStorage( storage );
	CvSeq* points = icvFastHessianDetector( sum, point_storage, quality );
      //  double t = (double)cvGetTickCount();
	CvSeq* descriptors = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvSURFDescriptor), storage );
	int dx_s[] = {2, 0, 0, 2, 4, 0, -1, 2, 0, 4, 4, 0, 1};
	int dy_s[] = {2, 0, 0, 4, 2, 0, -1, 0, 2, 4, 4, 0, 1};
	int dx_t[] = {2, 0, 0, 2, 4, 0, -1, 2, 0, 4, 4, 0, 1};
	int dy_t[] = {2, 0, 0, 4, 2, 0, -1, 0, 2, 4, 4, 0, 1};
	double x[81], *iter_x;
	double y[81], *iter_y;
	double angle[81], *iter_angle;
	double sumx, sumy;
	double temp_mod;
	int angle_n;
	for ( int kk = 0; kk < points->total; kk++ )
	{
	//	printf("%d\n",points->total);
		CvSURFPointOne* point = (CvSURFPointOne*)cvGetSeqElem( points, kk );
		CvSURFDescriptor descriptor;
		descriptor.x = cvRound(point->x);
		descriptor.y = cvRound(point->y);
		descriptor.laplacian = point->laplacian;
		int size = point->size;
		int layer = point->octave*FilterScale+point->scale;
		descriptor.s = size*1.2/9.;
		descriptor.mod = 0;
		/* repeatable orientation */
		iter_x = x;
		iter_y = y;
		iter_angle = angle;
		angle_n = 0;
		icvResizeHaarPattern( dx_s, dx_t, 9, size );
		icvResizeHaarPattern( dy_s, dy_t, 9, size );
		int* sum_ptr = (int*)sum->data.ptr;
		double c2 = 1./(descriptor.s*descriptor.s*2.5*2.5*2);
		double over_exp = 1./(3.14159*2*descriptor.s*descriptor.s*2.5*2.5);
		for ( int j = -6; j <=2; j++ )
		{
			int y = descriptor.y+j*size/9;
			if (( y >= 0 )&&( y < sum->rows-size ))
			{
				double ry = j+2;
				for ( int i = -6; i <=2; i++ )
				{
					int x = descriptor.x+i*size/9;
					if (( x >= 0 )&&( x < sum->cols-size ))
					{
						double rx = j+2;
						double radius = rx*rx+ry*ry;
						if ( radius <= 16 )
						{
							rx*=descriptor.s;
							ry*=descriptor.s;
							*iter_x = icvCalHaarPattern( sum_ptr+x+y*sum->cols, dx_t, sum->cols )*exp(-radius*c2)*over_exp;
							*iter_y = icvCalHaarPattern( sum_ptr+x+y*sum->cols, dy_t, sum->cols )*exp(-radius*c2)*over_exp;
							*iter_angle = cvFastArctan( *iter_y, *iter_x );
							iter_x++;
							iter_y++;
							iter_angle++;
							angle_n++;
						}
					}
				}
			}
		}
		double bestx = 0;
		double besty = 0;
		for ( int i = 0; i < 360; i+=5 )
		{
			sumx = 0;
			sumy = 0;
			iter_x = x;
			iter_y = y;
			iter_angle = angle;
			for ( int j = 0; j < angle_n; j++ )
			{
				if ( ( ( *iter_angle < i+60 )&&( *iter_angle > i ) )||
				( ( (*iter_angle+360) < i+60 )&&( (*iter_angle+360) > i ) ) )
				{
					sumx+=*iter_x;
					sumy+=*iter_y;
				}
				iter_x++;
				iter_y++;
				iter_angle++;
			}
			temp_mod = sumx*sumx+sumy*sumy;
			if ( temp_mod > descriptor.mod )
			{
				descriptor.mod = temp_mod;
				bestx = sumx;
				besty = sumy;
			}
		}
		descriptor.dir = cvFastArctan( besty, bestx );
		/* get sub-region (CV_INTER_AREA approximately retain the information of total image for haar feature while reduce the time consuming */
		double cos_dir = CosCache[MAX(cvRound(descriptor.dir*10)+3600, 0)%3600];
		double sin_dir = SinCache[MAX(cvRound(descriptor.dir*10)+3600, 0)%3600];
		cvmSet( wrap, 0, 0, cos_dir );
		cvmSet( wrap, 0, 1, -sin_dir );
		cvmSet( wrap, 0, 2, descriptor.x );
		cvmSet( wrap, 1, 0, sin_dir );
		cvmSet( wrap, 1, 1, cos_dir );
		cvmSet( wrap, 1, 2, descriptor.y );

		cvGetQuadrangleSubPix( img, regions_cache[layer], wrap );
		cvResize( regions_cache[layer], region_cache, CV_INTER_AREA );
		uchar* region_d;
		int region_step;
		cvGetImageRawData( region_cache, &region_d, &region_step );
		uchar* region_x = region_d+1;
		uchar* region_y = region_d+region_step;
		uchar* region_xy = region_d+1+region_step;
		region_step-=20;
		double* iter_dx = (double*)dx_cache->data.ptr;
		double* iter_dy = (double*)dy_cache->data.ptr;
		for ( int i = 0; i < 20; i++ )
		{
			for ( int j = 0; j < 20; j++ )
			{
				*iter_dx = *region_y-*region_d-*region_x+*region_xy;
				*iter_dy = *region_x-*region_d-*region_y+*region_xy;
				iter_dx++;
				iter_dy++;
				region_d++;
				region_x++;
				region_y++;
				region_xy++;
			}
			region_d+=region_step;
			region_x+=region_step;
			region_y+=region_step;
			region_xy+=region_step;
		}
		cvMul( gauss_kernel_cache, dx_cache, dx_cache );
		cvMul( gauss_kernel_cache, dy_cache, dy_cache );
		
		double tx, ty;
		double* iter_vector = descriptor.vector;
		if ( flags&CV_SURF_EXTENDED )
		{
			/* 128-bin descriptor */
			for ( int i = 0; i < 4; i++ )
				for ( int j = 0; j < 4; j++ )
				{
					iter_vector[0] = 0;
					iter_vector[1] = 0;
					iter_vector[2] = 0;
					iter_vector[3] = 0;
					iter_vector[4] = 0;
					iter_vector[5] = 0;
					iter_vector[6] = 0;
					iter_vector[7] = 0;
					for ( int x = i*5; x < i*5+5; x++ )
					{
						for ( int y = j*5; y < j*5+5; y++ )
						{
							tx = cvGetReal2D( dx_cache, x, y );
							ty = cvGetReal2D( dy_cache, x, y );
							if ( ty >= 0 )
							{
								iter_vector[0] += tx;
								iter_vector[1] += fabs(tx);
							} else {
								iter_vector[2] += tx;
								iter_vector[3] += fabs(tx);
							}
							if ( tx >= 0 )
							{
								iter_vector[4] += ty;
								iter_vector[5] += fabs(ty);
							} else {
								iter_vector[6] += ty;
								iter_vector[7] += fabs(ty);
							}
						}
					}
					/* unit vector is essential for contrast invariant */
					double normalize = 0;
					for ( int k = 0; k < 8; k++ )
						normalize+=iter_vector[k]*iter_vector[k];
					normalize = sqrt(normalize);
					for ( int k = 0; k < 8; k++ )
						iter_vector[k] = iter_vector[k]/normalize;
					iter_vector+=8;
				}
		} else {
			/* 64-bin descriptor */
			for ( int i = 0; i < 4; i++ )
				for ( int j = 0; j < 4; j++ )
				{
					iter_vector[0] = 0;
					iter_vector[1] = 0;
					iter_vector[2] = 0;
					iter_vector[3] = 0;
					for ( int x = i*5; x < i*5+5; x++ )
					{
						for ( int y = j*5; y < j*5+5; y++ )
						{
							tx = cvGetReal2D( dx_cache, x, y );
							ty = cvGetReal2D( dy_cache, x, y );
							iter_vector[0] += tx;
							iter_vector[1] += ty;
							iter_vector[2] += fabs(tx);
							iter_vector[3] += fabs(ty);
						}
					}
					double normalize = 0;
					for ( int k = 0; k < 4; k++ )
						normalize+=iter_vector[k]*iter_vector[k];
					normalize = sqrt(normalize);
					for ( int k = 0; k < 4; k++ )
						iter_vector[k] = iter_vector[k]/normalize;
					iter_vector+=4;
				}
		}

		cvSeqPush( descriptors, &descriptor );
	}
	cvReleaseMemStorage( &point_storage );
	cvReleaseMat( &sum );
      //  t = (double)cvGetTickCount()-t;
      //  printf( "descriptor time = %gms\n", t/((double)cvGetTickFrequency()*1000.) );
	return descriptors;
}

inline
double
icvCompareSURFDescriptor( CvSURFDescriptor* descriptor1,
			  CvSURFDescriptor* descriptor2,
			  double best,
			  int length = 64 )
{
	double* iter_vector1 = descriptor1->vector;
	double* iter_vector2 = descriptor2->vector;
	double total_cost = 0;
	for ( int i = 0; i < length; i++ )
	{
		total_cost+=(*iter_vector1-*iter_vector2)*(*iter_vector1-*iter_vector2);
		if ( total_cost > best )
			break;
		iter_vector1++;
		iter_vector2++;
	}
	return total_cost;
}

inline int
icvNaiveNearestNeighbor( CvSURFDescriptor* descriptor,
			 CvSeq* model_descriptors,
			 int length )
{
	int neighbor = -1;
	double d;
	double dist1 = 0xffffff, dist2 = 0xffffff;
	for ( int i = 0; i < model_descriptors->total; i++ )
	{
		CvSURFDescriptor* model_descriptor = (CvSURFDescriptor*)cvGetSeqElem( model_descriptors, i );
		if ( descriptor->laplacian != model_descriptor->laplacian )
			continue;
		d = icvCompareSURFDescriptor( descriptor, model_descriptor, dist2, length );
		if ( d < dist1 )
		{
			dist2 = dist1;
			dist1 = d;
			neighbor = i;
		} else {
			if ( d < dist2 )
				dist2 = d;
		}
	}
	if ( dist1 < 0.8*dist2 )
		return neighbor;
	return -1;
}


CvSeq*
cvSURFFindPair( CvSeq* ImageDescriptor,
		CvSeq* ObjectDescriptor,
		CvMemStorage* storage,
		int flags )
{

	CvSeq* correspond = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvSURFDescriptor), storage );
	int length = ( flags&CV_SURF_EXTENDED ) ? 128 : 64;
	int chaos = 0;
	for ( int i = 0; i < ObjectDescriptor->total; i++ )
	{
		CvSURFDescriptor* descriptor = (CvSURFDescriptor*)cvGetSeqElem( ObjectDescriptor, i );
		int nearest_neighbor = icvNaiveNearestNeighbor( descriptor, ImageDescriptor, length );
		if ( nearest_neighbor >= 0 )
		{
			cvSeqPush( correspond, descriptor);
			cvSeqPush( correspond, (CvSURFDescriptor*)cvGetSeqElem( ImageDescriptor, nearest_neighbor ) );
			chaos += nearest_neighbor;
		} 
	}  
	//printf("%d\n",correspond->total);
	return correspond;
	
}
