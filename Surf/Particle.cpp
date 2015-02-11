
#include"Particle.h"
#include "surf.h"
#include "cv.h"
#include "highgui.h"
#include<math.h>


//#include "stdafx.h"
#include <fstream>
#include <string>
using namespace std;

/*****************************************************************\
            Implementation of Particle Filter to track object           
\*****************************************************************/

CParticleFilterTrack::CParticleFilterTrack()
{
    IsInit     = false;
	m_pRGB     = NULL;
	m_pHSV     = NULL;
    m_pSURF    = NULL;
	m_wid      = 0;
	m_hei      = 0;
    m_object   = cvRect(0, 0, 0, 0);

    m_PF       = cvCreateParticleFilter(4,4,SAMPLENUM); // NO. of samples 4维 4维 50个粒子
    CvMat Dyn  = cvMat(4,4,CV_MAT4x4_32F,m_PF->DynamMatr);//4行4列矩阵

	/* |2  0  -1 0 | xk   //
	// |0  2  0  -1| yk   //
	// |1  0  0  0 | xk-1 //
	// |0  1  0  0 | yk-1 */
    /*A color-based tracking by Kalman particle filter - icpr04*/
    storage = cvCreateMemStorage(0);

    cvmSetZero(&Dyn);
    cvmSet(&Dyn, 0, 0, 2);//设置矩阵 0行 0列 值为2
	cvmSet(&Dyn, 0, 2, -1);
    cvmSet(&Dyn, 1, 1, 2);
    cvmSet(&Dyn, 1, 3, -1);
    cvmSet(&Dyn, 2, 0, 1);
    cvmSet(&Dyn, 3, 1, 1);

	confidences = NULL;
	surfConfidences = hsvConfidences = NULL;
	maxIdx = 0;
}

CParticleFilterTrack::~CParticleFilterTrack()
{
	cvReleaseParticleFilter(&m_PF);

	if(m_pHSV)
		cvReleaseImage(&m_pHSV);
	if (confidences) { delete[] confidences; confidences = NULL; }
	if (surfConfidences) {delete[] surfConfidences; surfConfidences = NULL; }
	if (hsvConfidences) {delete[] hsvConfidences; hsvConfidences = NULL; }
}

/*parameters instruction: rect is the location of the object in rgb*/
void CParticleFilterTrack::InitTracker(CvRect rect, IplImage * rgb)
{
	m_object       =  rect;//m_object
	m_pRGB         =  rgb;
	m_wid          =  rgb->width;
	m_hei          =  rgb->height;

    // initialize histogram
	IplROI  roi;
	if(m_pHSV)
		cvReleaseImage(&m_pHSV);
	m_pHSV = cvCreateImage(cvSize(m_wid, m_hei), IPL_DEPTH_8U, 3);//创建图m_pHSV rgb大小 深度 通道数

    m_pSURF = cvCreateImage(cvSize(m_wid, m_hei), IPL_DEPTH_8U, 1); //单通道灰度图像
	cvCvtColor(m_pRGB, m_pHSV, CV_BGR2HSV);//色彩空间转换到hsv
    cvCvtColor(m_pRGB,m_pSURF,CV_RGB2GRAY);//rgb ->gray

	roi.coi     = 0;//roi区域
    roi.xOffset = m_object.x;//m_object  CvRect：左上角点的坐标，目标的，高
	roi.yOffset = m_object.y;
	roi.height  = m_object.height;
	roi.width   = m_object.width;
    m_pHSV->roi = &roi;
    m_pSURF->roi = &roi;
	//calculate the Descriptor of the model
	//?? problem !!
// 	printf("%d\n",m_pSURF->nChannels);
// 	cvNamedWindow("hello",1);
// 	cvShowImage("hello",m_pSURF);
// 	cvWaitKey(0);
    ModelDescriptors = cvSURFDescriptor( m_pSURF, storage, 4., CV_SURF_EXTENDED );

    printf("Object Descriptors: %d\n", ModelDescriptors->total);
    CalcROIHSVHisto(m_pHSV, m_histo);//初始化HSV颜色直方图
	m_pHSV->roi = NULL;
	m_pSURF->roi = NULL;
	// the end

	LBound[0] = (float)m_object.x-0.1f*m_object.width;
	LBound[1] = (float)m_object.y-0.1f*m_object.height;
	LBound[2] = 0;
	LBound[3] = 0;

	UBound[0] = (float)m_object.x + 1.1f*m_object.width;
	UBound[1] = (float)m_object.y + 1.1f*m_object.height;
	UBound[2] = 0;
	UBound[3] = 0;
	
	CvMat LB = cvMat(4,1,CV_MAT4x1_32F,LBound);//4行1列矩阵，边界值
	CvMat UB = cvMat(4,1,CV_MAT4x1_32F,UBound);
	//cvParticleFilterInitSampleSet(m_PF,&LB,&UB);//初始化粒子

	IsInit = true;//标志位置为1
}

//void CParticleFilterTrack::Disconnect()//取消跟踪目标
//{
//	m_pRGB         = NULL;
//	m_wid          = 0;
//	m_hei          = 0;
//    m_object       = cvRect(0, 0, 0, 0);	
//
//	IsInit = false;
//}


//compute normalized histogram of value-normalized(0-1) HSV
//Please refer to "Color Based Probabilistic Tracking -- ECCV02"
bool CParticleFilterTrack::CalcROIHSVHisto(IplImage * hsv_src, HISTO  histo_dst)
{   
	if(NULL == hsv_src || NULL == histo_dst)  return false;
    if(NULL == hsv_src->imageData) return false;
	if(NULL == hsv_src->roi) return false;

	int width, height; 
	int widthstep, yindex, xindex;

	unsigned char ucHue, ucSat, ucValue;
	float hue, sat, value;//hue, saturation, and value of image
	
	int i, j; //temporary variable
    int index; //index of histogram bins
    
    for(i = 0; i < HSV_BIN_NUM; i++)
		histo_dst[i] = 0;
    
	width     = hsv_src->roi->width;//ROI区域
	height    = hsv_src->roi->height;
    widthstep = hsv_src->widthStep;

	for(i=0; i < height; i++)
	{
		for(j=0; j < width; j++)//eccv
		{
			yindex = i+hsv_src->roi->yOffset;
			xindex = j+hsv_src->roi->xOffset;
			
			ucHue   =  (unsigned char)hsv_src->imageData[ yindex*widthstep + 3*xindex];
			ucSat   =  (unsigned char)hsv_src->imageData[ yindex*widthstep + 3*xindex +1];
			ucValue =  (unsigned char)hsv_src->imageData[ yindex*widthstep + 3*xindex +2];

			hue   = float( ucHue / 255.);
			sat   = float( ucSat / 255.);
			value = float( ucValue / 255.);
			
            if(sat >= SAT_THRE && value >= VAL_THRE)
			{
				index = int( HSV_H_BIN_NUM * ( ucHue == 255 ? HSV_H_BIN_NUM-1 : floor(HSV_H_BIN_NUM*hue) ) +
					    ( ucSat == 255 ? HSV_S_BIN_NUM-1 : floor( HSV_S_BIN_NUM * (sat-SAT_THRE)/(1-SAT_THRE) ) ) );
				//compute index of histogram bins
				histo_dst[index]++;
			}
			else
			{
				index = int( HSV_H_BIN_NUM*HSV_S_BIN_NUM + 
					    ( ucValue == 255 ? HSV_V_BIN_NUM-1 : floor(HSV_V_BIN_NUM*value) ) );
				//compute index of histogram bins
				histo_dst[index]++;
			}

		}
	}
    //normalize histogram
    for(i=0; i<HSV_BIN_NUM; i++)
		histo_dst[i] /= (width*height);

	return 1;  
}

void CParticleFilterTrack::classify(Patches* trackingPatches)
{
	int i, j, top, left, wid, hei;
	HISTO histo;
	float dist;
	float sum = 0.f;
	//surf
	float sumSurf = 0.f;
	float sum2 = 0.f;
	IplROI roi;

	cvCvtColor(m_pRGB,m_pSURF,CV_RGB2GRAY);//rgb ->gray
	cvCvtColor(m_pRGB, m_pHSV, CV_BGR2HSV);
	m_pHSV->roi = &roi;//roi非空
	m_pSURF->roi = &roi;
	wid = m_object.width;
	hei = m_object.height;

	int patchNum = trackingPatches->getNum();
	if (confidences == NULL) confidences = new float[patchNum];
	if (surfConfidences == NULL) surfConfidences = new float[patchNum];
	if (hsvConfidences == NULL) hsvConfidences = new float[patchNum];
	float maxConfidence = 0;
	for (i = 0; i < patchNum; i++)
	{
		top = trackingPatches->getRect(i).upper;
		left = trackingPatches->getRect(i).left;

		if(top < 0 || left < 0 || top+hei > m_hei || left+wid > m_wid)//超出图像宽高度
		{
			confidences[i] = 0;
			continue;
		}

		roi.coi     = 0;
		roi.xOffset = left;
		roi.yOffset = top;
		roi.height  = hei;
		roi.width   = wid;

		//calculate the Descriptor of the new-patch
		if(ModelDescriptors->total == 0)
		{
			//matchSurf[i] = 0;
			surfConfidences[i] = 0;
		}
		else
		{
			ImageDescriptors = cvSURFDescriptor( m_pSURF, storage, 4., CV_SURF_EXTENDED );
			if(ImageDescriptors->total == 0)
			{
			//matchSurf[i] = 0;
				surfConfidences[i] = 0;
			}
			else
			{
			 seq = cvSURFFindPair( ImageDescriptors, ModelDescriptors, storage, CV_SURF_EXTENDED );
			 //printf("%d	",seq->total);
			 //matchSurf[i] = (float)seq->total;
			 surfConfidences[i] = (float)seq->total;
				
			}
		}
	
		sumSurf += matchSurf[i];
			//	printf("%d	",sumSurf);
        //----------------------------

		CalcROIHSVHisto(m_pHSV, histo);

        dist = .0;
		for(j = 0; j < HSV_BIN_NUM; j++)
			dist += (float)sqrt(m_histo[j]*histo[j]);//m_histo即初始的直方图
		dist = 1 - dist;

		hsvConfidences[i] = (float)exp(-20*dist);//初步分配权值 eccv公式（7）
		sum += hsvConfidences[i];
	}
		
	//normalize sample weight
	
	if(sumSurf == 0.f)
		for( i = 0; i < patchNum; i++)//权值归一
		{
			confidences[i] = hsvConfidences[i] / sum;
			if (confidences[i] > maxConfidence)
			{
				maxConfidence = confidences[i];
				maxIdx = i;
			}
		}
	else
	{
		for( i = 0; i < patchNum; i++)//权值分配
		{
			hsvConfidences[i] /= sum;
			surfConfidences[i] /= sumSurf;
			confidences[i] = COEFF * hsvConfidences[i] + (1-COEFF) * surfConfidences[i];
			if (confidences[i] > maxConfidence)
			{
				maxConfidence = confidences[i];
				maxIdx = i;
			}
		}
	}
	trackedPatch = trackingPatches->getRect(maxIdx);
	m_pHSV->roi = NULL;//不只针对roi
	m_pSURF->roi = NULL;

}

void CParticleFilterTrack::update(Rect trackedResult)
{
	m_object.x = trackedResult.left;
	m_object.y = trackedResult.upper;
}

//void CParticleFilterTrack::MyCPDHisto(CvParticleFilter * PF)//直方图 粒子权值
//{
//	int i, j, top, left, wid, hei;
//	HISTO histo;
//	float dist;
//	float sum = 0.f;
//	//surf
//	float sumSurf = 0.f;
//    float sum2 = 0.f;
//	IplROI roi;
//   // m_pSURF = cvCreateImage(cvSize(m_wid, m_hei), IPL_DEPTH_8U, 1); //单通道灰度图像
//    cvCvtColor(m_pRGB,m_pSURF,CV_RGB2GRAY);//rgb ->gray
//	cvCvtColor(m_pRGB, m_pHSV, CV_BGR2HSV);
//    m_pHSV->roi = &roi;//roi非空
//    m_pSURF->roi = &roi;
//	wid = m_object.width;
//	hei = m_object.height;
//
//	for(i = 0; i < PF->SamplesNum; i++)//对于每个粒子
//	{
//		top  = int(PF->flSamples[i][1]-0.5*hei);//y坐标值
//		left = int(PF->flSamples[i][0]-0.5*wid);//x坐标
//
//		if(top < 0 || left < 0 || top+hei > m_hei || left+wid > m_wid)//超出图像宽高度
//		{
//			PF->flConfidence[i] = 0;
//			continue;
//		}
//
//		roi.coi     = 0;
//		roi.xOffset = left;
//		roi.yOffset = top;
//		roi.height  = hei;
//	    roi.width   = wid;
//
//		//calculate the Descriptor of the new-patch
//		if(ModelDescriptors->total == 0)
//		{
//			matchSurf[i] = 0;
//		}
//		else
//		{
//			ImageDescriptors = cvSURFDescriptor( m_pSURF, storage, 4., CV_SURF_EXTENDED );
//			if(ImageDescriptors->total == 0)
//			{
//			matchSurf[i] = 0;
//			}
//			else
//			{
//			 seq = cvSURFFindPair( ImageDescriptors, ModelDescriptors, storage, CV_SURF_EXTENDED );
//			 printf("%d	",seq->total);
//			 matchSurf[i] = (float)seq->total;
//				
//			}
//		}
//	
//		sumSurf += matchSurf[i];
//			//	printf("%d	",sumSurf);
//        //----------------------------
//
//		CalcROIHSVHisto(m_pHSV, histo);
//
//        dist = .0;
//		for(j = 0; j < HSV_BIN_NUM; j++)
//			dist += (float)sqrt(m_histo[j]*histo[j]);//m_histo即初始的直方图
//		dist = 1 - dist;
//
//		PF->flConfidence[i] = (float)exp(-20*dist);//初步分配权值 eccv公式（7）
//		sum += PF->flConfidence[i];
//	}
//
//	//normalize sample weight
//	if(sumSurf == 0.f)
//    	for( i = 0; i < PF->SamplesNum; i++)//权值归一
//		{
//		PF->flConfidence[i] /= sum;
//		}
//    else
//	{
//        for( i = 0; i < PF->SamplesNum; i++)//权值分配
//		{
//	 		PF->flConfidence[i] /= sum;
//			matchSurf[i] /= sumSurf;
//			PF->flConfidence[i] = COEFF*PF->flConfidence[i] + (1-COEFF)*matchSurf[i];
//			sum2 += PF->flConfidence[i];
//		}
//		for( i = 0; i < PF->SamplesNum; i++)//权值归一
//		{
//			PF->flConfidence[i] /= sum2;
//		}
//	}
//	m_pHSV->roi = NULL;//不只针对roi
//	m_pSURF->roi = NULL;
//}
//
//void CParticleFilterTrack::Running()//此处将原程序改过了
//{
//    int top, left, right, bottom;
//	int i;
//    float Neff = 0.f;
//    IplROI  roi;
// 	cvParticleFilterPropagate(m_PF);
//	MyCPDHisto(m_PF);
//    
//    DrawRect(CV_RGB(255,255,0));//画出n个粒子的矩形框 255,255,0
//	
//	// compute the effective sample size
//	for( i = 0; i < m_PF->SamplesNum; i++)
//	{
//		Neff += (m_PF->flConfidence[i] * m_PF->flConfidence[i]);
//	}
//
//	Neff = 1.f/Neff;
//	
////	if( Neff < m_PF->SamplesNum/2)        // 越小说明粒子退化越严重
//	
//
//	 cvParticleFilterExpectation(m_PF);//此处将输出程序改过了
// 
//	//correct
//	
//	m_object.x = cvRound( m_PF->State[0]-m_object.width*0.5);//通过舍入将浮点数转为整数
//	m_object.y = cvRound( m_PF->State[1]-m_object.height*0.5 );
//	
//	//show tracking result
//	left   = m_object.x;
//	top    = m_object.y;
//	right  = m_object.x + m_object.width-1;
//	bottom = m_object.y + m_object.height-1;
//
//	top--, left--, bottom++, right++;
//	
//	if(top < 0) top = 0;
//	if(left < 0) left = 0;
//	if(bottom > m_hei-1) bottom = m_hei-1;
//	if(right >  m_wid-1) right  = m_wid-1;
//	
//	//这里可以对尺度空间模板进行更新
//    cvCvtColor(m_pRGB,m_pSURF,CV_RGB2GRAY);//rgb ->gray
//    roi.coi = 0;
//    roi.xOffset = left;
//    roi.yOffset = top;
//    roi.height = (int)abs(top-bottom)+8;
//    roi.width = (int)abs(left-right)+8;
//    m_pSURF->roi = &roi;
//    ModelDescriptors = cvSURFDescriptor( m_pSURF, storage, 4., CV_SURF_EXTENDED );
//
////   if(ModelBackup->total > 1)
////	    ModelDescriptors = ModelBackup;
//
//  printf("Object Descriptors: %d\n", ModelDescriptors->total);
//   m_pSURF->roi = NULL; //更新之后将roi置空
//
//	//对角线两点绘制矩形
//    cvRectangle(m_pRGB, cvPoint(left, top), cvPoint(right, bottom), CV_RGB(255,0,0), 1);//255,0,0
// printf("zhongxindian: %d   %d\n", right-(right-left)/2,bottom-(bottom-top)/2);
//
//
///*	FILE *fp = fopen("E:\\num.txt", "a+");
//
////	float x=right-(right-left)/2;
//	float y=bottom-(bottom-top)/2;
////	string s1="x:";
////	string s2="y:";
//	string hh="\x000D\x000A";
////	fputs(s1.c_str(), fp);
////	char buf[20];
/////	itoa(x, buf, 10); 
////	string buff=buf;
////	fputs(buff.c_str(),fp);
//	fputs(hh.c_str(),fp);
////	fputs(s2.c_str(), fp);
//	char buf2[20];
//    itoa(y, buf2, 10); 
//	string buff2=buf2;
//	fputs(buff2.c_str(),fp);
//	fputs(hh.c_str(),fp);
//	fseek( fp, 0l, SEEK_END ); 
//
//	fclose(fp);
//
//*/
//
//    cvParticleFilterResample(m_PF);	
//}
//
//
//CvRect CParticleFilterTrack::GetWindow()
//{
//	return m_object;
//}
//
//
//void CParticleFilterTrack::DrawRect(CvScalar color)
//{
//	int i;
//	int top, left, right, bottom, hei, wid;
//
//	hei = m_object.height;
//	wid = m_object.width;
//
//	for(i = 0; i < m_PF->SamplesNum; i++)
//	{
//    	top    = int(m_PF->flSamples[i][1] - 0.5f*hei);
//		left   = int(m_PF->flSamples[i][0] - 0.5f*wid);
//		bottom = top+hei-1;
//        right  = left+wid-1;
//
//		top--, left--, bottom++, right++;
//
//        if(top < 0) top = 0;
//		if(left < 0) left = 0;
//		if(bottom > m_hei-1) bottom = m_hei-1;
//		if(right >  m_wid-1) right  = m_wid-1;
//
//		cvRectangle(m_pRGB, cvPoint(left, top), cvPoint(right, bottom), color, 1);	
//
//	}
//}
/*---------------------------------------------------------------*\
                  The end of class CParticleFilterTrack           
\*---------------------------------------------------------------*/
