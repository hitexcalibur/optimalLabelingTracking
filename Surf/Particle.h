#ifndef _PARTICLE_H_
#define _PARTICLE_H_

#include "cv.h"
//#include "global.h"
#include ".\Filter.h"
#include "..\Online Boosting\framework\OnlineBoosting\Patches.h"

#define COEFF 0.55f
#define SAMPLENUM 50

#define HSV_H_BIN_NUM 10
#define HSV_S_BIN_NUM 10
#define HSV_V_BIN_NUM 10
#define HSV_BIN_NUM  HSV_H_BIN_NUM * HSV_S_BIN_NUM + HSV_V_BIN_NUM
#define SAT_THRE 0.1  //the threshold of saturation  仅仅使用H、S值超过阈值的像素点用于直方图计算
#define VAL_THRE 0.2  //the threshold of value 


typedef float HISTO[HSV_BIN_NUM]; //definition of histogram (10H*10S+10V)

class CParticleFilterTrack
{
	
public:
    
	CParticleFilterTrack();
	~CParticleFilterTrack();
	
	void InitTracker(CvRect rect, IplImage * rgb);
	//void Disconnect();
	//void Running();
	//CvRect GetWindow();

	void classify(Patches* trackingPatches);
	void update(Rect trackedResult);
	float* confidences;
	float* surfConfidences;
	float* hsvConfidences;
	int maxIdx;
	Rect trackedPatch;

	bool IsInit;
	IplImage * m_pRGB;
	//surf to storage the points
	CvMemStorage* storage;
	CvSeq* ModelDescriptors;   //to storage the model IPts
	CvSeq* ImageDescriptors;  //to storage the new-patch IPts
    CvSeq* seq; //to storage the match point
	CvSeq* ModelBackup;//haha
	float matchSurf[SAMPLENUM];
private:
	
	//void MyCPDHisto(CvParticleFilter * PF);
	bool CalcROIHSVHisto(IplImage * hsv_src, HISTO  histo_dst);
	//void DrawRect(CvScalar color);
	
	//frame property
	//IplImage * m_pRGB;
	IplImage * m_pHSV;
	IplImage * m_pSURF;
	int m_hei;
	int m_wid;
	
	float Dynam[16];
	float LBound[4];
	float UBound[4];
	
	CvRect  m_object;
	HISTO   m_histo;
	CvParticleFilter * m_PF;
};//Particle

#endif