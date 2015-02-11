// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License 
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

#include "OnlineBoost.h"
#include "Tracker.h"
#include "Public.h"
#include "Sample.h"

BoostingTrackerParams::BoostingTrackerParams()
{
	numBaseClassifier = 100;
	overlap = 0.99;
	searchFactor = 2;
}

CvHaarClassifierCascade* Tracker::facecascade = NULL;

SimpleTracker::SimpleTracker()
{ 
	facecascade = NULL; 
	m_MILlabels = NULL; 
	m_BoostingTrackinglabels = NULL; 
	m_SemiBoostingTrackinglabels = NULL;
	m_BeyongdSemiBoostingTrackinglabels = NULL;
	m_labels = NULL;

	trackerNum = 6;   //6
	validJudgeThershold = trackerNum * 0.7;
	trackedValid = NULL;
	MILconfidences = NULL;
	m_wholetrackedLost = false;
	_fragTracker = NULL;

	priorTracker = new float[trackerNum];
	for (int i = 0; i < trackerNum; i++)
		priorTracker[i] = 1;
	trackedValid = new bool[trackerNum];


	combineOut.open("D:\\Study\\VIlab\\Experiment\\bear1\\result\\combineOut.txt");
	MILout.open("D:\\Study\\VIlab\\Experiment\\bear1\\result\\MILout.txt");
	OnlineOut.open("D:\\Study\\VIlab\\Experiment\\bear1\\result\\OnlineOut.txt");
	SemiOut.open("D:\\Study\\VIlab\\Experiment\\bear1\\result\\SemiOut.txt");
	BeyondOut.open("D:\\Study\\VIlab\\Experiment\\bear1\\result\\BeyondOut.txt");
	SurfOut.open("D:\\Study\\VIlab\\Experiment\\bear1\\result\\SurfOut.txt");
	FragOut.open("D:\\Study\\VIlab\\Experiment\\bear1\\result\\FragOut.txt");
	trackerCondidencOut.open("D:\\Study\\VIlab\\Experiment\\bear1\\result\\Tracker.txt");
}

SimpleTracker::~SimpleTracker()
{ 
	if( _clf!=NULL ) delete _clf;  
	if (m_MILlabels!=NULL) delete[] m_MILlabels; m_MILlabels = NULL;
	if (m_BoostingTrackinglabels != NULL) delete[] m_BoostingTrackinglabels; m_BoostingTrackinglabels = NULL;
	if (m_SemiBoostingTrackinglabels != NULL) delete[] m_SemiBoostingTrackinglabels; m_SemiBoostingTrackinglabels = NULL;
	if (m_BeyongdSemiBoostingTrackinglabels != NULL) delete[] m_BeyongdSemiBoostingTrackinglabels; m_BeyongdSemiBoostingTrackinglabels = NULL;
	if (m_labels != NULL) delete[trackerNum] m_labels; m_labels = NULL;
	if (trackedValid != NULL) delete[] trackedValid; trackedValid = NULL;
	if (MILconfidences != NULL) delete[] MILconfidences;
	if (priorTracker != NULL) delete[] priorTracker; priorTracker = NULL;

	combineOut.close();
	MILout.close();
	OnlineOut.close();
	SemiOut.close();
	BeyondOut.close();
	SurfOut.close();
	FragOut.close();
	trackerCondidencOut.close();
}

bool			SimpleTracker::init(Matrixu frame, SimpleTrackerParams p, ClfStrongParams *clfparams)
{
	static Matrixu *img;

	img = &frame;
	frame.initII();

	_clf = ClfStrong::makeClf(clfparams);
	_curState.resize(4);
	for(int i=0;i<4;i++ ) _curState[i] = p._initstate[i];
	SampleSet posx, negx;

	fprintf(stderr,"Initializing Tracker..\n");

	// sample positives and negatives from first frame
	posx.sampleImage(img, (uint)_curState[0],(uint)_curState[1], (uint)_curState[2], (uint)_curState[3], p._init_postrainrad);
	negx.sampleImage(img, (uint)_curState[0],(uint)_curState[1], (uint)_curState[2], (uint)_curState[3], 2.0f*p._srchwinsz, (1.5f*p._init_postrainrad), p._init_negnumtrain);
	if( posx.size()<1 || negx.size()<1 ) return false;

	// train
	_clf->update(posx,negx);
	negx.clear();

	img->FreeII();

	_trparams = p;
	_clfparams = clfparams;
	_cnt = 0;
	return true;
}
void			SimpleTracker::track_frames(vector<Matrixu> &video, SimpleTrackerParams p, BoostingTrackerParams btparams, ClfStrongParams *clfparams)
{
	CvVideoWriter* w = NULL;
	Matrixu states(video.size(), 4);
	Matrixu t; string pnd="#";
	uchar* curBackground;

	// save video file
	if( ! p._vidsave.empty() ){
		w = cvCreateVideoWriter( p._vidsave.c_str(), CV_FOURCC('I','Y','U','V'), 15, cvSize(video[0].cols(), video[0].rows()), 3 );
		if( w==NULL ) abortError(__LINE__,__FILE__,"Error opening video file for output");
	}

	// initialization
	 frameind = 0;
	if( p._initWithFace ){  // init with face
		fprintf(stderr,"Searching for face...\n");
		while( !Tracker::initFace(&p,video[frameind]) ){
			video[frameind].conv2RGB(t); t._keepIpl=true;
			t.drawText(("#"+int2str(frameind,3)).c_str(),1,25,255,255,0);
			// display on screen & write to disk
			if( p._disp ){ t.display(1); cvWaitKey(1); }
			if( w != NULL ) cvWriteFrame( w, t.getIpl() );
			t._keepIpl=false; t.freeIpl();
			//frameind++;
		}
		clfparams->_ftrParams->_width	= (uint)p._initstate[2];
		clfparams->_ftrParams->_height	= (uint)p._initstate[3];
		init(video[frameind], p, clfparams);
		
	} // init with params
	else{
		clfparams->_ftrParams->_width	= (uint)p._initstate[2];
		clfparams->_ftrParams->_height	= (uint)p._initstate[3];
		init(video[0], p, clfparams);
		states(frameind,0) = (uint)_curState[0];
		states(frameind,1) = (uint)_curState[1];
		states(frameind,2) = (uint)_curState[2];
		states(frameind,3) = (uint)_curState[3];
		
		//init BoostingTracker
		Rect trackingRect;
		unsigned char *curFrame=NULL;
		trackingRect = btparams.initBB;
		m_trackedResult = btparams.initBB;

		video[frameind].createIpl();

		IplImage *srcImg = video[frameind].getIpl();
		IplImage *dstImg = cvCreateImage( cvGetSize(srcImg), srcImg->depth, 1 );
		cvCvtColor( srcImg, dstImg, CV_BGR2GRAY );
		curFrame = reinterpret_cast<unsigned char*>(dstImg->imageData);

		Size frameSize;
		frameSize.height = video[0].rows();
		frameSize.width = video[0].cols();

		ImageRepresentation* curFrameRep = new ImageRepresentation(curFrame, frameSize); 
		Rect wholeImage;
		wholeImage = frameSize;

		printf ("init Boosting Tracker...");
		_boostingTracker = new BoostingTracker (curFrameRep, trackingRect, wholeImage, btparams.numBaseClassifier);

		//init SemiBoostingTracker
		_semiBoostingTracker = new SemiBoostingTracker(curFrameRep, trackingRect, wholeImage, btparams.numBaseClassifier);
		printf (" done.\n");

		//init BeyondSemiBoostingTracker

		IplImage* cropped=cvCreateImage(cvSize(trackingRect.width, trackingRect.height), IPL_DEPTH_8U, 1);
		cvSetImageROI(dstImg, cvRect(0, 0,trackingRect.width, trackingRect.height));
		cvCopy(dstImg, cropped);
		cvSetImageROI(dstImg, cvRect(trackingRect.left, trackingRect.upper,trackingRect.width, trackingRect.height));
		cvCopy(cropped, dstImg);
		cvSetImageROI(dstImg, cvRect(0, 0,wholeImage.width, wholeImage.height));
		curBackground=(uchar*) dstImg->imageData;
		
		ImageRepresentation* curBGMRep;

		curBGMRep=new ImageRepresentation(curBackground, frameSize);

		_beyondSemiBoostingTracker = new BeyondSemiBoostingTracker(curFrameRep, curBGMRep, trackingRect, wholeImage, btparams.numBaseClassifier);

		//init FragTracker
		_fragParameters.initial_tl_x = btparams.initBB.left;
		_fragParameters.initial_tl_y = btparams.initBB.upper;
		_fragParameters.initial_br_x = btparams.initBB.left + btparams.initBB.width - 1;
		_fragParameters.initial_br_y = btparams.initBB.upper + btparams.initBB.height - 1;
		_fragParameters.search_margin = 7;                       //equal to the _trparams._srchwinsz 
		_fragParameters.B = 16;
		_fragParameters.metric_used = 3;
		CvMat* curr_img = cvCreateMat(srcImg->height, srcImg->width, CV_8U);
		cvCopy(dstImg, curr_img);
		_fragTracker = new Fragments_Tracker(curr_img, _fragParameters);
		_fragTracker->trackedPatch = btparams.initBB;
		_fragTracker->params = &_fragParameters;
		cvReleaseMat(&curr_img);

		//init surfTracker
		CvRect rect = cvRect(btparams.initBB.left, btparams.initBB.upper, btparams.initBB.width, btparams.initBB.height);
		cvSURFInitialize();
		_surfTracker.InitTracker(rect, srcImg);

		video[frameind].freeIpl();
		//if (cropped != NULL) {
		//	cvReleaseImage(&cropped); cropped = NULL;
		//}
		//if (dstImg != NULL) {
		//	cvReleaseImage(&dstImg); dstImg = NULL;
		//}

		//frameind++;
	}

	// track rest of frames
	StopWatch sw(true); double ttt;
	for( frameind; frameind<(int)video.size(); frameind++ )
	{
		ttt = sw.Elapsed(true);
		fprintf(stderr,"%s%d Frames/%f sec = %f FPS",ERASELINE,frameind,ttt,((double)frameind)/ttt);
		track_frame(video[frameind],t, curBackground);
		cout << "Frame No." << frameind << " x:" << m_trackedResult.left << " y:" << m_trackedResult.upper << "\n";
		cout << "Tracker Confidence: ";
		for (int i = 0; i < trackerNum; i++) {
			trackerCondidencOut << priorTracker[i] << " ";
			cout << priorTracker[i] << " ";
		}
		trackerCondidencOut << "\n";
		cout << endl;
		if( p._disp || w!= NULL ){
			video[frameind] = t;
			video[frameind].drawText(("#"+int2str(frameind,3)).c_str(),1,25,255,255,0);
			video[frameind].createIpl();
			video[frameind]._keepIpl = true;
			// display on screen
			//if( p._disp ){
				video[frameind].display(1);
				cvWaitKey(1);
			//}
			// save video
			if( w != NULL && frameind<(int)video.size()-1 )
				Matrixu::WriteFrame(w, video[frameind]);

			//save image to file
			char imageName[100];
			char imageFormat[100];
			sprintf_s(imageFormat,"img%05i.bmp", frameind);
			sprintf_s(imageName,"D:\\Study\\VIlab\\Experiment\\bear1\\result\\%s", imageFormat);
			cvFlip(video[frameind].getIpl(), video[frameind].getIpl(), 0);
			cvSaveImage(imageName, video[frameind].getIpl());
			cvFlip(video[frameind].getIpl(), video[frameind].getIpl(), 0);

			video[frameind]._keepIpl = false;
			video[frameind].freeIpl();
		}
		
		for( int s=0; s<4; s++ ) states(frameind,s) = (uint)_curState[s];

	}

	// save states
	if( !p._trsave.empty() ){
		bool scs = states.DLMWrite(p._trsave.c_str());
		if( !scs ) abortError(__LINE__,__FILE__,"error saving states to file");
	}

	// clean up
	if( w != NULL )
		cvReleaseVideoWriter( &w );

}
double			SimpleTracker::track_frame(Matrixu &frame, Matrixu &framedisp, uchar* curBackground)
{
	static SampleSet posx, negx, detectx;
	static vectorf prob;
	static vectori order;
	static Matrixu *img;

	int imageNum;

	double resp;

	IplImage *srcImg = NULL, *dstImg = NULL;
	frame.createIpl();
	srcImg = frame.getIpl();

	// copy a color version into framedisp (this is where we will draw a colored box around the object for output)
	frame.conv2RGB(framedisp);

	img = &frame;
	frame.initII();

	// run current clf on search window
	//if (!m_wholetrackedLost)
	//{
		detectx.sampleImage(img,(uint)_curState[0],(uint)_curState[1],(uint)_curState[2],(uint)_curState[3], (float)_trparams._srchwinsz);
	//}
	//else
	//{
		//detectx.sampleImage(img,(uint)_curState[0],(uint)_curState[1],(uint)_curState[2],(uint)_curState[3], max(frame.rows(), frame.cols()));
	//}
	prob = _clf->classify(detectx,_trparams._useLogR);

	imageNum = detectx.size();

    MILconfidences = new float[imageNum];
	for (int i = 0; i < imageNum; i++)
		MILconfidences[i] = prob[i];
	// find best location
	int bestind = max_idx(prob);
	resp=prob[bestind];
	MILmaxIdx = bestind;

	_curState[1] = (float)detectx[bestind]._row; 
	_curState[0] = (float)detectx[bestind]._col;

	//run Boosting Tracker clf
	//prepare image info

	unsigned char* curFrame = NULL;
	ImageRepresentation* curFrameRep;
	dstImg = cvCreateImage( cvGetSize(srcImg), srcImg->depth, 1 );
	cvCvtColor( srcImg, dstImg, CV_BGR2GRAY );
	curFrame = reinterpret_cast<unsigned char*>(dstImg->imageData);
	


	ImageRepresentation* curBGMRep;

	//trans sampleset to patches
	Patches *trackingPatches = new PatchesRegularScan(detectx);	
	Rect searchRegion;
	Size frameSize = Size(frame.rows(), frame.cols());

	curFrameRep = new ImageRepresentation(curFrame, frameSize);
	searchRegion = trackingPatches->getROI();
	curFrameRep->setNewImageAndROI(curFrame, searchRegion);

	curBGMRep = new ImageRepresentation(curBackground, frameSize);
	curBGMRep->setNewImageAndROI(curBackground, searchRegion);

	


	_boostingTracker->classify(curFrameRep, trackingPatches);
	_semiBoostingTracker->classify(curFrameRep, trackingPatches);
	_beyondSemiBoostingTracker->classify(curFrameRep, curBGMRep, trackingPatches);

	//fragTracker
	CvMat *curr_img = cvCreateMat(srcImg->height, srcImg->width, CV_8U);
	cvCopy(dstImg, curr_img);
	_fragTracker->compute_IH(curr_img, _fragTracker->IIV_I);

	vector<int> x_coords;
	vector<int> y_coords;

	int new_yM, new_xM;
	double score_M;

	_fragTracker->find_template(_fragTracker->template_patches_histograms,
		_fragTracker->patches,
		curr_img->height, curr_img->width,
		_fragTracker->curr_pos_y - (_fragTracker->params->search_margin),
		_fragTracker->curr_pos_x - (_fragTracker->params->search_margin),
		_fragTracker->curr_pos_y + (_fragTracker->params->search_margin),
		_fragTracker->curr_pos_x + (_fragTracker->params->search_margin),
		new_yM, new_xM, score_M,
		x_coords, y_coords);
	_fragTracker->Update_Template(_fragTracker->curr_template->height,_fragTracker->curr_template->width,new_yM,new_xM,1,curr_img);

	_surfTracker.m_pRGB = srcImg;
	_surfTracker.classify(trackingPatches);

	//output the result to files
	MILout << _curState[0] << " " << _curState[1] << " " << _curState[2] << " " << _curState[3] << "\n";
	OnlineOut << _boostingTracker->trackedPatch.left << " " << _boostingTracker->trackedPatch.upper << " " << _boostingTracker->trackedPatch.width << " " << _boostingTracker->trackedPatch.height << "\n";
	SemiOut << _semiBoostingTracker->trackedPatch.left << " " << _semiBoostingTracker->trackedPatch.upper << " " << _semiBoostingTracker->trackedPatch.width << " " << _semiBoostingTracker->trackedPatch.height << "\n";
	BeyondOut << _beyondSemiBoostingTracker->trackedPatch.left << " " << _beyondSemiBoostingTracker->trackedPatch.upper << " " << _beyondSemiBoostingTracker->trackedPatch.width << " " << _beyondSemiBoostingTracker->trackedPatch.height << "\n";
	SurfOut << _surfTracker.trackedPatch.left << " " << _surfTracker.trackedPatch.upper << " " << _surfTracker.trackedPatch.width << " " << _surfTracker.trackedPatch.height << "\n";
	FragOut << _fragTracker->curr_template_tl_x << " " << _fragTracker->curr_template_tl_y << " " << _fragTracker->curr_template_width << " " << _fragTracker->curr_template_height << "\n";


	//display the tracking result of all trackers
	framedisp.drawRect(_curState[2], _curState[3], _curState[0], _curState[1], 1, 0, 3, 255, 255, 0 );
	//boosting tracker display
	framedisp.drawRect((float)_boostingTracker->trackedPatch.width, (float)_boostingTracker->trackedPatch.height, (float)_boostingTracker->trackedPatch.left, (float)_boostingTracker->trackedPatch.upper, 1, 0, 3, 255, 0, 255);
	//semi boosting tracker display
	framedisp.drawRect((float)_semiBoostingTracker->trackedPatch.width, (float)_semiBoostingTracker->trackedPatch.height, (float)_semiBoostingTracker->trackedPatch.left, (float)_boostingTracker->trackedPatch.upper, 1, 0, 3, 0, 255, 0);
	//beyond semi boosting tracker display
	framedisp.drawRect((float)_beyondSemiBoostingTracker->trackedPatch.width, (float)_beyondSemiBoostingTracker->trackedPatch.height, (float)_beyondSemiBoostingTracker->trackedPatch.left, (float)_beyondSemiBoostingTracker->trackedPatch.upper, 1, 0, 3, 0, 0, 255);
	//fragTracker display
	framedisp.drawRect((float)_fragTracker->curr_template_width, (float)_fragTracker->curr_template_height, (float)_fragTracker->curr_template_tl_x, (float)_fragTracker->curr_template_tl_y, 1, 0, 3, 0, 255, 255);
	//surfTracker display
	framedisp.drawRect(_surfTracker.trackedPatch.width, _surfTracker.trackedPatch.height, _surfTracker.trackedPatch.left, _surfTracker.trackedPatch.upper, 1, 0, 3, 200, 150, 200);

	for (int i = 0; i < trackerNum; i++)
		trackedValid[i] = true;

	//VersionA
	//prepareLabels(imageNum);
	//int resultIdx = calculateTrackingResult(imageNum, trackingPatches);

	//VersionB
	int resultIdx = calculateTrackingResult_versionB(trackingPatches, detectx);

	m_trackedResult = trackingPatches->getRect(resultIdx);
	calculatePrior();

	//Tracker's own update
	//_boostingTracker->update(curFrameRep, trackingPatches, trackingPatches->getRect(_boostingTracker->detector->getPatchIdxOfBestDetection()));
	//_semiBoostingTracker->update(curFrameRep, trackingPatches, trackingPatches->getRect(_semiBoostingTracker->detector->getPatchIdxOfBestDetection()));
	//_beyondSemiBoostingTracker->myupdate(curFrameRep, curBGMRep, trackingPatches, _beyondSemiBoostingTracker->trackedPatch);
	//backgroundModel(curFrame, curBackground, srcImg->width, srcImg->height, _beyondSemiBoostingTracker->trackedPatch);

	

	_boostingTracker->update(curFrameRep, trackingPatches, m_trackedResult);
	_semiBoostingTracker->update(curFrameRep, trackingPatches, m_trackedResult);
	_beyondSemiBoostingTracker->myupdate(curFrameRep, curBGMRep, trackingPatches, m_trackedResult);
	backgroundModel(curFrame, curBackground, srcImg->width, srcImg->height, m_trackedResult);
	_fragTracker->Update_Template(_fragTracker->curr_template->height,_fragTracker->curr_template->width,m_trackedResult.upper + m_trackedResult.height / 2, m_trackedResult.left + m_trackedResult.width / 2,1,curr_img);

	_curState[1] = (float)m_trackedResult.upper;
	_curState[0] = (float)m_trackedResult.left;


	/////// DEBUG /////// display actual probability map
	if( _trparams._debugv ){
		Matrixf probimg(frame.rows(),frame.cols());
		for( uint k=0; k<(uint)detectx.size(); k++ )
			probimg(detectx[k]._row, detectx[k]._col) = prob[k];

		probimg.convert2img().display(2,2);
		cvWaitKey(1);
	}
	

	// train location clf (negx are randomly selected from image, posx is just the current tracker location)

	if( _trparams._negsamplestrat == 0 )
		negx.sampleImage(img, _trparams._negnumtrain, (int)_curState[2], (int)_curState[3]);
	else
		negx.sampleImage(img, (int)_curState[0], (int)_curState[1], (int)_curState[2], (int)_curState[3], 
			(1.5f*_trparams._srchwinsz), _trparams._posradtrain+5, _trparams._negnumtrain);

	if( _trparams._posradtrain == 1 )
		posx.push_back(img, (int)_curState[0], (int)_curState[1], (int)_curState[2], (int)_curState[3]);
	else
		posx.sampleImage(img, (int)_curState[0], (int)_curState[1], (int)_curState[2], (int)_curState[3], _trparams._posradtrain, 0, _trparams._posmaxtrain);

	_clf->update(posx,negx);

	/////// DEBUG /////// display sampled negative points
	if( _trparams._debugv ){
		for( int j=0; j<negx.size(); j++ )
			framedisp.drawEllipse(1,1,(float)negx[j]._col,(float)negx[j]._row,1,255,0,255);
	}

	// draw a colored box around object

	combineOut << m_trackedResult.left << " " << m_trackedResult.upper << " " << m_trackedResult.width << " " << m_trackedResult.height << "\n";
	framedisp.drawRect(m_trackedResult.width, m_trackedResult.height, m_trackedResult.left, m_trackedResult.upper, 1, 0, 3, 255, 0, 0);
	m_wholetrackedLost = false;



	// clean up
	img->FreeII();
	posx.clear(); negx.clear(); detectx.clear();

	
	cvReleaseMat(&curr_img);
	cvReleaseImage(&dstImg);

	frame.freeIpl();
	_cnt++;


	x_coords.clear();
	y_coords.clear();

	if (MILconfidences != NULL)
	{
		delete[] MILconfidences;
		MILconfidences = NULL;
	}

	if (_surfTracker.confidences != NULL)
	{
		delete[] _surfTracker.confidences;
		_surfTracker.confidences = NULL;
	}

	if (_surfTracker.surfConfidences != NULL)
	{
		delete[] _surfTracker.surfConfidences;
		_surfTracker.surfConfidences = NULL;
	}
	
	if (_surfTracker.hsvConfidences != NULL)
	{
		delete[] _surfTracker.hsvConfidences;
		_surfTracker.hsvConfidences = NULL;
	}

	return resp;
}

void SimpleTracker::validJudge()
{
	int i, j, k;
	int *index;
	float *temp;

	temp = new float[trackerNum];
	index = new int[trackerNum];


	for (k = 0; k < trackerNum; k++)
		if (priorTracker[k] != 0) break;

	if (k < trackerNum)
	{
	for (i = 0; i < trackerNum; i++)
	{	
		temp[i] = priorTracker[i];
		index[i] = 0;
	}

	for (i = 0; i < trackerNum; i++)
	{
		int maxidx = i;
		for (j = 0; j < trackerNum; j++)
			if (temp[j] > temp[maxidx]) maxidx = j;
		index[i] = maxidx;
		temp[maxidx] = 0;
	}
	float sum = 0;
	for (i = 0; i < trackerNum; i++)
	{
		sum += priorTracker[index[i]];
		if (sum > validJudgeThershold) break;
	}
	for (j = i + 1; j < trackerNum; j++)
		trackedValid[j] = false;
	}
	delete[] temp;
	delete[] index;

}

int SimpleTracker::calculateTrackingResult_versionB(Patches* trackingPatches, SampleSet sampleSet)
{
	Dataset data;

	int i, j;

	int numLabelers = trackerNum;

	int patchesNum = trackingPatches->getNum();


	//if (_boostingTracker->trackedLost) { trackedValid[1] = false; numLabelers--; }
	//if (_semiBoostingTracker->trackedLost) { trackedValid[2] = false; numLabelers--; }
	//if (_beyondSemiBoostingTracker->trackedLost) { trackedValid[3] = false; numLabelers--; }

	data.numLabelers = numLabelers;

	int labelId = 0;
	int imageID = 0;

	vector<Label> vectorLabel;

	for (i = 0; i < patchesNum; i++)
	{
		if (i == MILmaxIdx || i == _boostingTracker->detector->getPatchIdxOfBestDetection() || i == _semiBoostingTracker->detector->getPatchIdxOfBestDetection() || i == _beyondSemiBoostingTracker->m_maxIdx || i == _fragTracker->resultIdx)
		{
			Rect tempRect = trackingPatches->getRect(i);
			
			if (trackedValid[0])          //MIL
			{
				Label tempLabel;
				tempLabel.imageIdx = imageID;
				tempLabel.labelerId = 0;
				tempLabel.patchIdx = i;

				float tempDis = sqrt((_curState[0] - tempRect.left) * (_curState[0] - tempRect.left) + (_curState[1] - tempRect.upper) * (_curState[1] - tempRect.upper));

				if (i == MILmaxIdx || tempDis < DIS_THRESHOLD || inTopTen(i, MILconfidences, patchesNum)) tempLabel.label = 1;
				else tempLabel.label = 0;
				vectorLabel.push_back(tempLabel);
				labelId++;
			}
			if (trackedValid[1])            //Online Boosting
			{
				Label tempLabel;
				tempLabel.imageIdx = imageID;
				tempLabel.labelerId = 1;
				tempLabel.patchIdx = i;

				float tempDis = sqrt(float(_boostingTracker->getTrackedPatch().left - tempRect.left) * (_boostingTracker->getTrackedPatch().left - tempRect.left) + (_boostingTracker->getTrackedPatch().upper - tempRect.upper) * (_boostingTracker->getTrackedPatch().upper - tempRect.upper));

				if (i == _boostingTracker->detector->getPatchIdxOfBestDetection() || tempDis < DIS_THRESHOLD || inTopTen(i, _boostingTracker->detector->getConfidences(), patchesNum)) tempLabel.label = 1;
				else tempLabel.label = 0;

				vectorLabel.push_back(tempLabel);
				labelId++;
			}
			if (trackedValid[2])                //SemiBoosting
			{
				Label tempLabel;
				tempLabel.imageIdx = imageID;
				tempLabel.labelerId = 2;
				tempLabel.patchIdx = i;

				float tempDis = sqrt(float(_semiBoostingTracker->getTrackedPatch().left - tempRect.left) * (_semiBoostingTracker->getTrackedPatch().left - tempRect.left) + (_semiBoostingTracker->getTrackedPatch().upper - tempRect.upper) * (_semiBoostingTracker->getTrackedPatch().upper - tempRect.upper));

				if (i == _semiBoostingTracker->detector->getPatchIdxOfBestDetection() || tempDis < DIS_THRESHOLD || inTopTen(i, _semiBoostingTracker->detector->getConfidences(), patchesNum)) tempLabel.label = 1;
				else tempLabel.label = 0;

				vectorLabel.push_back(tempLabel);
				labelId++;
			}
			if (trackedValid[3])             //BeyondSemiBoosting
			{
				Label tempLabel;

				tempLabel.imageIdx = imageID;
				tempLabel.labelerId = 3;
				tempLabel.patchIdx = i;

				float tempDis = sqrt(float(_beyondSemiBoostingTracker->getTrackedPatch().left - tempRect.left) * (_beyondSemiBoostingTracker->getTrackedPatch().left - tempRect.left) + (_beyondSemiBoostingTracker->getTrackedPatch().upper - tempRect.upper) * (_beyondSemiBoostingTracker->getTrackedPatch().upper - tempRect.upper));

				if (i == _beyondSemiBoostingTracker->m_maxIdx || tempDis < DIS_THRESHOLD || inTopTen(i, _beyondSemiBoostingTracker->m_confidences, patchesNum)) tempLabel.label = 1;
				else tempLabel.label = 0;

				vectorLabel.push_back(tempLabel);
				labelId++;
			}
			if (trackedValid[4])            //Surf
			{
				Label tempLabel;

				tempLabel.imageIdx = imageID;
				tempLabel.labelerId = 4;
				tempLabel.patchIdx = i;

				float tempDis = sqrt(float(_surfTracker.trackedPatch.left - tempRect.left) * (_surfTracker.trackedPatch.left - tempRect.left) + (_surfTracker.trackedPatch.upper - tempRect.upper) * (_surfTracker.trackedPatch.upper - tempRect.upper));

				if (i == _surfTracker.maxIdx || tempDis < DIS_THRESHOLD || inTopTen(i, _surfTracker.confidences, patchesNum)) tempLabel.label = 1;
				else tempLabel.label = 0;
				vectorLabel.push_back(tempLabel);
				labelId++;

			}
			if (trackedValid[5])               //Frag
			{
				Label tempLabel;

				tempLabel.imageIdx = imageID;
				tempLabel.labelerId = 5;
				tempLabel.patchIdx = i;

				float tempDis = sqrt(float(_fragTracker->curr_template_tl_x - tempRect.left) * (_fragTracker->curr_template_tl_x - tempRect.left) + (_fragTracker->curr_template_tl_y - tempRect.upper) * (_fragTracker->curr_template_tl_y - tempRect.upper));

				if (i == _fragTracker->resultIdx || tempDis < DIS_THRESHOLD || inTopTen(i, _fragTracker->m_confidences, patchesNum)) tempLabel.label = 1;
				else tempLabel.label = 0;
				vectorLabel.push_back(tempLabel);
				labelId++;
			}

			imageID++;
		}
	}

	data.numImages = imageID;
	data.numLabels = labelId;

	data.priorZ1 = 0.5;

	data.priorAlpha = (double *) malloc(sizeof(double) * data.numLabelers);

	int tempId = 0;
	//printf("Assuming prior on alpha has mean 1 and std 1\n");
	//for (i = 0; i < data.numLabelers; i++) {
	//	//data.priorAlpha[i] = 1.0; /* default value */
	//	if (trackedValid[tempId])
	//	{
	//		data.priorAlpha[i] = priorTracker[tempId];
	//		tempId++;
	//	}
	//	else tempId++;
	//}
	for (i = 0; i <data.numLabelers; i++)
		data.priorAlpha[i] = priorTracker[i];

	data.priorBeta = (double *) malloc(sizeof(double) * data.numImages);

	//printf("Assuming prior on beta has mean 1 and std 1\n");
	for (j = 0; j < data.numImages; j++) {
		data.priorBeta[j] = 1.0; /* default value */
	}
	data.probZ1 = (double *) malloc(sizeof(double) * data.numImages);
	data.probZ0 = (double *) malloc(sizeof(double) * data.numImages);
	data.beta = (double *) malloc(sizeof(double) * data.numImages);
	data.alpha = (double *) malloc(sizeof(double) * data.numLabelers);
	data.labels = (Label *) malloc(sizeof(Label) * data.numLabels);

	for (i = 0; i < data.numLabels; i++)
		data.labels[i] = vectorLabel[i];

	vectorLabel.clear();

	EM(&data);

	double maxProb = 0;
	int maxIdx = -1;
	
	for (i = 0; i < data.numImages; i++)
	{
		if (data.probZ1[i] > maxProb)
		{
			maxProb = data.probZ1[i];
			maxIdx = i;
		}
	}

	int maxPatchesID;
	for (i = 0; i < data.numLabels; i++)
		if (data.labels[i].imageIdx == maxIdx)
		{
			maxPatchesID = data.labels[i].patchIdx;
			break;
		}

	free(data.priorAlpha);
	free(data.priorBeta);
	free(data.labels);
	free(data.alpha);
	free(data.beta);
	free(data.probZ1);
	free(data.probZ0);

	return maxPatchesID;

}

bool SimpleTracker::inTopTen(int idx, float* confidence, int imageNum)
{
	int cal = 0;
	for (int i = 0; i < imageNum; i++)
	{
		if (confidence[i] > confidence[idx]) cal++;
		if (cal > 10) break;
	}
	if (cal <= 10) return true;
	else return false;

}


void SimpleTracker::calculatePrior()
{
	float tempDis;

	tempDis = sqrt((m_trackedResult.left - _curState[0]) * (m_trackedResult.left - _curState[0]) + (m_trackedResult.upper - _curState[1]) * (m_trackedResult.upper - _curState[1]));
	if (tempDis > DIS_THRESHOLD || (!trackedValid[0])) priorTracker[0] = (1 - TRACKERADJUST) * priorTracker[0];
	else priorTracker[0] = (1 - TRACKERADJUST) * (priorTracker[0]) + TRACKERADJUST;

	tempDis = sqrt(float(m_trackedResult.left - _boostingTracker->getTrackedPatch().left) * (m_trackedResult.left - _boostingTracker->getTrackedPatch().left) + (m_trackedResult.upper - _boostingTracker->getTrackedPatch().upper) * (m_trackedResult.upper - _boostingTracker->getTrackedPatch().upper));
	if (tempDis > DIS_THRESHOLD || (!trackedValid[1])) priorTracker[1] = (1 - TRACKERADJUST) * priorTracker[1];
	else priorTracker[1] = (1 - TRACKERADJUST) * (priorTracker[1]) + TRACKERADJUST;

	tempDis = sqrt(float(m_trackedResult.left - _semiBoostingTracker->getTrackedPatch().left) * (m_trackedResult.left - _semiBoostingTracker->getTrackedPatch().left) + (m_trackedResult.upper - _semiBoostingTracker->getTrackedPatch().upper) * (m_trackedResult.upper - _semiBoostingTracker->getTrackedPatch().upper));
	if (tempDis > DIS_THRESHOLD || (!trackedValid[2])) priorTracker[2] = (1 - TRACKERADJUST) * priorTracker[2];
	else priorTracker[2] = (1 - TRACKERADJUST) * (priorTracker[2]) + TRACKERADJUST;

	tempDis = sqrt(float(m_trackedResult.left - _beyondSemiBoostingTracker->getTrackedPatch().left) * (m_trackedResult.left - _beyondSemiBoostingTracker->getTrackedPatch().left) + (m_trackedResult.upper - _beyondSemiBoostingTracker->getTrackedPatch().upper) * (m_trackedResult.upper - _beyondSemiBoostingTracker->getTrackedPatch().upper));
	if (tempDis > DIS_THRESHOLD || (!trackedValid[3])) priorTracker[3] = (1 - TRACKERADJUST) * priorTracker[3];
	else priorTracker[3] = (1 - TRACKERADJUST) * (priorTracker[3]) + TRACKERADJUST;

	tempDis = sqrt(float(m_trackedResult.left - _surfTracker.trackedPatch.left) * (m_trackedResult.left - _surfTracker.trackedPatch.left) + (m_trackedResult.upper - _surfTracker.trackedPatch.upper) * (m_trackedResult.upper - _surfTracker.trackedPatch.upper));
	if (tempDis > DIS_THRESHOLD || (!trackedValid[4])) priorTracker[4] = (1 - TRACKERADJUST) * priorTracker[4];
	else priorTracker[4] = (1 - TRACKERADJUST) * (priorTracker[4]) + TRACKERADJUST;

	tempDis = sqrt(float(m_trackedResult.left - _fragTracker->curr_template_tl_x) * (m_trackedResult.left - _fragTracker->curr_template_tl_x) + (m_trackedResult.upper - _fragTracker->curr_template_tl_y) * (m_trackedResult.upper - _fragTracker->curr_template_tl_y));
	if (tempDis > DIS_THRESHOLD || (!trackedValid[5])) priorTracker[5] = (1 - TRACKERADJUST) * priorTracker[5];
	else priorTracker[5] = (1 - TRACKERADJUST) * (priorTracker[5]) + TRACKERADJUST;

}

void SimpleTracker::prepareLabels(int imageNum)
{
	int i;

	if (_boostingTracker->trackedLost) trackedValid[1] = false;
	if (_semiBoostingTracker->trackedLost) trackedValid[2] = false;
	if (_beyondSemiBoostingTracker->trackedLost) trackedValid[3] = false;

	m_labels = new int*[trackerNum];
	m_MILlabels = calculateLabels(MILconfidences, imageNum); m_labels[0] = m_MILlabels;
	if (trackedValid[1]) 
	{
		m_BoostingTrackinglabels = calculateLabels(_boostingTracker->detector->getConfidences(), imageNum);
		m_labels[1] = m_BoostingTrackinglabels;
	}else m_labels[1] = NULL;

	if (trackedValid[2])
	{
		m_SemiBoostingTrackinglabels = calculateLabels(_semiBoostingTracker->detector->getConfidences(), imageNum);
		m_labels[2] = m_SemiBoostingTrackinglabels;
	}else m_labels[2] = NULL;

	if (trackedValid[3])
	{
		m_BeyongdSemiBoostingTrackinglabels = calculateLabels(_beyondSemiBoostingTracker->m_confidences, imageNum);
		m_labels[3] = m_BeyongdSemiBoostingTrackinglabels;
	}else m_labels[3] = NULL;
	

}

int* SimpleTracker::calculateLabels(float* confidences, int imageNum)
{
	int *labels = NULL;
	labels = new int[imageNum];

	int exampleNum = 10;

	int *maxExample = NULL;
	int *minExample = NULL;

	maxExample = new int[exampleNum];
	minExample = new int[exampleNum];
	
	int maxidx;

	int i, j;

	if (labels != NULL)
	{
		for (i = 0; i < exampleNum; i++)
			maxExample[i] = minExample[i] = i;
		for (i = 1; i < imageNum; i++)
		{
			labels[i] = 0;
			for (j = 0; j < exampleNum; j++)
			{
				if (confidences[i] > confidences[maxExample[j]]) {
					maxExample[j] = i;
					break;
				}
				if (confidences[i] < confidences[minExample[j]]) {
					minExample[j] = i;
					break;
				}
			}
		}
		for (i = 0; i < exampleNum; i++)
		{
			labels[maxExample[i]] = 1;
			labels[minExample[i]] = -1;
		}
	}

	delete[] maxExample;
	delete[] minExample;
	return labels;

}

int SimpleTracker::calculateTrackingResult(int imageNum, Patches* patches)
{
	Dataset data;
	int i, j;

	int numLabelers = 0;
	int numLabels = 0;

	for (i = 0; i < trackerNum; i++)
		if (trackedValid[i]) numLabelers++;

	int idx = 0;
	for (i = 0; i < imageNum; i++)
		for (j = 0; j < trackerNum; j++)
		{
			if (trackedValid[j]) 
				if (m_labels[j][i] == 1 || m_labels[j][i] == -1) numLabels++;
		}

	data.numLabels = numLabels;
	data.numLabelers = numLabelers;
	data.numImages = imageNum;
	data.priorZ1 = 0.5;

	data.priorAlpha = (double *) malloc(sizeof(double) * data.numLabelers);

	int tempId = 0;
	//printf("Assuming prior on alpha has mean 1 and std 1\n");
	for (i = 0; i < data.numLabelers; i++) {
		//data.priorAlpha[i] = 1.0; /* default value */
		if (trackedValid[tempId])
		{
			data.priorAlpha[i] = priorTracker[tempId];
			tempId++;
		}
		else tempId++;
	}
	data.priorBeta = (double *) malloc(sizeof(double) * data.numImages);

	//printf("Assuming prior on beta has mean 1 and std 1\n");
	for (j = 0; j < data.numImages; j++) {
		data.priorBeta[j] = 1.0; /* default value */
	}
	data.probZ1 = (double *) malloc(sizeof(double) * data.numImages);
	data.probZ0 = (double *) malloc(sizeof(double) * data.numImages);
	data.beta = (double *) malloc(sizeof(double) * data.numImages);
	data.alpha = (double *) malloc(sizeof(double) * data.numLabelers);
	data.labels = (Label *) malloc(sizeof(Label) * data.numLabels);

	for (i = 0; i < imageNum; i++)
	{
		for (j = 0; j < trackerNum; j++)
		{
			if (trackedValid[j])
			{
				if (m_labels[j][i] == 1) {
					data.labels[idx].label = 1;
					data.labels[idx].imageIdx = i;
					data.labels[idx].labelerId = j;
					idx++;
				}
				if (m_labels[j][i] == -1) {
					data.labels[idx].label = 0;
					data.labels[idx].imageIdx = i;
					data.labels[idx].labelerId = j;
					idx++;
				}

			}
		}
	}

	EM(&data);

	double maxProb = 0;
	int maxIdx = -1;
	float disWithPreResult = 0;

	for (i = 0; i < imageNum; i++)
		if (data.probZ1[i] >= maxProb)
		{
			Rect temp = patches->getRect(i);
			float tempDis = sqrt(float((temp.left-m_trackedResult.left)*(temp.left-m_trackedResult.left)+(temp.upper-m_trackedResult.upper)*(temp.upper-m_trackedResult.upper)));
			if (disWithPreResult == 0)
			{
				disWithPreResult = tempDis;
				maxProb = data.probZ1[i];
				maxIdx = i;
			}
			else if (tempDis < disWithPreResult)
			{
				tempDis = disWithPreResult;
				maxProb = data.probZ1[i];
				maxIdx = i;
			}
		
		}

	free(data.priorAlpha);
	free(data.priorBeta);
	free(data.labels);
	free(data.alpha);
	free(data.beta);
	free(data.probZ1);
	free(data.probZ0);

	return maxIdx;
}

void SimpleTracker::EM(Dataset *data)
{
	int i, j;
	const double THRESHOLD = 1E-5;
	double Q, lastQ;

	srand(time(NULL));

	/* Initialize parameters to starting values */
	for (i = 0; i < data->numLabelers; i++) {
		data->alpha[i] = data->priorAlpha[i];
		/*data->alpha[i] = (double) rand() / RAND_MAX * 4 - 2;*/
	}
	for (j = 0; j < data->numImages; j++) {
		data->beta[j] = data->priorBeta[j];
		/*data->beta[j] = (double) rand() / RAND_MAX * 3;*/
	}

	Q = 0;
	EStep(data);
	Q = computeQ(data);
	//printf("Q = %f\n", Q);
	do {
		lastQ = Q;

		/* Re-estimate P(Z|L,alpha,beta) */
		EStep(data);
		Q = computeQ(data);
		//printf("Q = %f; L = %f\n", Q, 0.0 /*computeLikelihood(data)*/);

		/*outputResults(data);*/
		MStep(data);

		Q = computeQ(data);
		//printf("Q = %f; L = %f\n", Q, 0.0 /*computeLikelihood(data)*/);
	} while (fabs((Q - lastQ)/lastQ) > THRESHOLD);
}


void SimpleTracker::backgroundModel(unsigned char* curFrameGray, unsigned char* BGMGray, int cols, int rows, Rect trackedRect)
{
	//simply update the image approximating the curFrame
	//except at the tracked or detected locations

	for (int r=0; r<rows; r++) {
		for (int c=0; c<cols; c++) {
			if (!((r>(trackedRect.upper) && r<(trackedRect.height+trackedRect.upper)) && 
				(c>(trackedRect.left)&& c<(trackedRect.width+trackedRect.left)))) {
					BGMGray[r*cols+c]=curFrameGray[r*cols+c];
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

double			SimpleTracker::track_frame(Matrixu &frame, Matrixu &framedisp)
{
	static SampleSet posx, negx, detectx;
	static vectorf prob;
	static vectori order;
	static Matrixu *img;

	double resp;

	// copy a color version into framedisp (this is where we will draw a colored box around the object for output)
	frame.conv2RGB(framedisp);

	img = &frame;
	frame.initII();

	// run current clf on search window
	detectx.sampleImage(img,(uint)_curState[0],(uint)_curState[1],(uint)_curState[2],(uint)_curState[3], (float)_trparams._srchwinsz);
	prob = _clf->classify(detectx,_trparams._useLogR);

	/////// DEBUG /////// display actual probability map
	if( _trparams._debugv ){
		Matrixf probimg(frame.rows(),frame.cols());
		for( uint k=0; k<(uint)detectx.size(); k++ )
			probimg(detectx[k]._row, detectx[k]._col) = prob[k];

		probimg.convert2img().display(2,2);
		cvWaitKey(1);
	}

	// find best location
	int bestind = max_idx(prob);
	resp=prob[bestind];

	_curState[1] = (float)detectx[bestind]._row; 
	_curState[0] = (float)detectx[bestind]._col;

	// train location clf (negx are randomly selected from image, posx is just the current tracker location)

	if( _trparams._negsamplestrat == 0 )
		negx.sampleImage(img, _trparams._negnumtrain, (int)_curState[2], (int)_curState[3]);
	else
		negx.sampleImage(img, (int)_curState[0], (int)_curState[1], (int)_curState[2], (int)_curState[3], 
		(1.5f*_trparams._srchwinsz), _trparams._posradtrain+5, _trparams._negnumtrain);

	if( _trparams._posradtrain == 1 )
		posx.push_back(img, (int)_curState[0], (int)_curState[1], (int)_curState[2], (int)_curState[3]);
	else
		posx.sampleImage(img, (int)_curState[0], (int)_curState[1], (int)_curState[2], (int)_curState[3], _trparams._posradtrain, 0, _trparams._posmaxtrain);

	_clf->update(posx,negx);

	/////// DEBUG /////// display sampled negative points
	if( _trparams._debugv ){
		for( int j=0; j<negx.size(); j++ )
			framedisp.drawEllipse(1,1,(float)negx[j]._col,(float)negx[j]._row,1,255,0,255);
	}

	// clean up
	img->FreeII();
	posx.clear(); negx.clear(); detectx.clear();

	// draw a colored box around object
	framedisp.drawRect(_curState[2], _curState[3], _curState[0], _curState[1], 1, 0,
		_trparams._lineWidth, _trparams._boxcolor[0], _trparams._boxcolor[1], _trparams._boxcolor[2] );

	_cnt++;

	return resp;
}


void			Tracker::replayTracker(vector<Matrixu> &vid, string statesfile, string outputvid, uint R, uint G, uint B)
{
	Matrixf states;
	states.DLMRead(statesfile.c_str());
	Matrixu colorframe;

	// save video file
	CvVideoWriter* w = NULL;
	if( ! outputvid.empty() ){
		w = cvCreateVideoWriter( outputvid.c_str(), CV_FOURCC('I','Y','U','V'), 15, cvSize(vid[0].cols(), vid[0].rows()), 3 );
		if( w==NULL ) abortError(__LINE__,__FILE__,"Error opening video file for output");
	}

	for( uint k=0; k<vid.size(); k++ )
	{	
		vid[k].conv2RGB(colorframe);
		colorframe.drawRect(states(k,2),states(k,3),states(k,0),states(k,1),1,0,2,R,G,B);
		colorframe.drawText(("#"+int2str(k,3)).c_str(),1,25,255,255,0);
		colorframe._keepIpl=true;
		colorframe.display(1,2);
		cvWaitKey(1);
		if( w != NULL )
			cvWriteFrame( w, colorframe.getIpl() );
		colorframe._keepIpl=false; colorframe.freeIpl();
	}

	// clean up
	if( w != NULL )
		cvReleaseVideoWriter( &w );
}
void			Tracker::replayTrackers(vector<Matrixu> &vid, vector<string> statesfile, string outputvid, Matrixu colors)
{
	Matrixu states;
	vector<Matrixu> resvid(vid.size());
	Matrixu colorframe;

	// save video file
	CvVideoWriter* w = NULL;
	if( ! outputvid.empty() ){
		w = cvCreateVideoWriter( outputvid.c_str(), CV_FOURCC('I','Y','U','V'), 15, cvSize(vid[0].cols(), vid[0].rows()), 3 );
		if( w==NULL ) abortError(__LINE__,__FILE__,"Error opening video file for output");
	}

	for( uint k=0; k<vid.size(); k++ ){
		vid[k].conv2RGB(resvid[k]);
		resvid[k].drawText(("#"+int2str(k,3)).c_str(),1,25,255,255,0);
	}

	for( uint j=0; j<statesfile.size(); j++ ){
		states.DLMRead(statesfile[j].c_str());
		for( uint k=0; k<vid.size(); k++ )	
			resvid[k].drawRect(states(k,3),states(k,2),states(k,0),states(k,1),1,0,3,colors(j,0),colors(j,1),colors(j,2));
	}

	for( uint k=0; k<vid.size(); k++ ){
		resvid[k]._keepIpl=true;
		resvid[k].display(1,2);
		cvWaitKey(1);
		if( w!=NULL && k<vid.size()-1)
			Matrixu::WriteFrame(w, resvid[k]);
		resvid[k]._keepIpl=false; resvid[k].freeIpl();
	}

	// clean up
	if( w != NULL )
		cvReleaseVideoWriter( &w );
}
bool			Tracker::initFace(TrackerParams* params, Matrixu &frame)
{
	const char* cascade_name = "haarcascade_frontalface_alt_tree.xml";
	const int minsz = 20;
	if( Tracker::facecascade == NULL )
		Tracker::facecascade = (CvHaarClassifierCascade*)cvLoad( cascade_name, 0, 0, 0 );

	frame.createIpl();
	IplImage *img = frame.getIpl();
	IplImage* gray = cvCreateImage( cvSize(img->width, img->height), IPL_DEPTH_8U, 1 );
    cvCvtColor(img, gray, CV_BGR2GRAY );
	frame.freeIpl();
	cvEqualizeHist(gray, gray);

	CvMemStorage* storage = cvCreateMemStorage(0);
	cvClearMemStorage(storage);
	CvSeq* faces = cvHaarDetectObjects(gray, Tracker::facecascade, storage, 1.05, 3, CV_HAAR_DO_CANNY_PRUNING ,cvSize(minsz, minsz));
	
	int index = faces->total-1;
	CvRect* r = (CvRect*)cvGetSeqElem( faces, index );
	
	

	while(r && (r->width<minsz || r->height<minsz || (r->y+r->height+10)>frame.rows() || (r->x+r->width)>frame.cols() ||
		r->y<0 || r->x<0)){
		r = (CvRect*)cvGetSeqElem( faces, --index);
	}

	//if( r == NULL ){
	//	cout << "ERROR: no face" << endl;
	//	return false;
	//}
	//else 
	//	cout << "Face Found: " << r->x << " " << r->y << " " << r->width << " " << r->height << endl;
	if( r==NULL )
		return false;

	//fprintf(stderr,"x=%f y=%f xmax=%f ymax=%f imgw=%f imgh=%f\n",(float)r->x,(float)r->y,(float)r->x+r->width,(float)r->y+r->height,(float)frame.cols(),(float)frame.rows());

	params->_initstate.resize(4);
	params->_initstate[0]	= (float)r->x;// - r->width;
	params->_initstate[1]	= (float)r->y;// - r->height;
	params->_initstate[2]	= (float)r->width;
	params->_initstate[3]	= (float)r->height+10;


	return true;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
				TrackerParams::TrackerParams()
{
	_boxcolor.resize(3);
	_boxcolor[0]	= 204;
	_boxcolor[1]	= 25;
	_boxcolor[2]	= 204;
	_lineWidth		= 2;
	_negnumtrain	= 15;
	_posradtrain	= 1;
	_posmaxtrain	= 100000;
	_init_negnumtrain = 1000;
	_init_postrainrad = 3;
	_initstate.resize(4);
	_debugv			= false;
	_useLogR		= true;
	_disp			= true;
	_initWithFace	= true;
	_vidsave		= "";
	_trsave			= "";
}

				SimpleTrackerParams::SimpleTrackerParams()
{
	_srchwinsz		= 30;
	_initstate.resize(4);
	_negsamplestrat	= 1;
}

