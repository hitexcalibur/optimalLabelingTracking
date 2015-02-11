// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License 
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

#include "Matrix.h"
#include "ImageFtr.h"
#include "Tracker.h"
#include "Public.h"

#include ".\Online Boosting\Boostiong Tracker\BoostingTracker.h"


void	Exp(const char* dir, const char* name, int strong, int trial, bool savevid=false);
void	FaceTrackDemo(const char* savefile);

void BoostingTrack(int numBaseClassifier, float overlap, float searchFactor, Rect initBB, vector<Matrixu> &vid);

int		main(int argc, char * argv[])
{
	string clipname;
	vector<string> clipnames;
	switch( 2 ){
		case 1: // FACE TRACK DEMO
			FaceTrackDemo(argc>1 ? argv[1] : 0);
			break;
		case 2: // RUN ON VIDEO
			if( argc<3 ) abortError(__LINE__,__FILE__,"Not enough parameters.  See README file.");
			Exp(argv[1],argv[2],(argc>3?atoi(argv[3]):1),(argc>4?atoi(argv[4]):1),(argc>5?atoi(argv[5])==1:0));
			break;					
		default:
			break;
	}

}
void	Exp(const char* dir, const char* name, int strong, int trial, bool savevid)
{
	bool success=true; 
	randinitalize(trial);

	string dataDir= string(dir);
	if( dataDir[dataDir.length()-2] != '/' ) dataDir+="/";
	dataDir += (string(name) + "/");

	Matrixf frameb, initstate;
	vector<Matrixu> vid;
	
	// read in frames and ground truth

	//added by bnzhong, for debuging
	printf("%s\n",(dataDir + name + "_frames.txt").c_str());
	//end of added by bnzhong, for debuging

	success=frameb.DLMRead((dataDir + name + "_frames.txt").c_str());
	if( !success ) abortError(__LINE__,__FILE__,"Error: frames file not found.");
	success=initstate.DLMRead((dataDir + name + "_gt.txt").c_str());
	if( !success ) abortError(__LINE__,__FILE__,"Error: gt file not found.");

	// TRACK

	vid.clear();
	/* commented by bnzhong
	parameter1:  image dirctroy
    parameter2:  image name
    parameter3:  image type
    parameter4:  image start frame number
    parameter5:  image end frame number
	parameter6:  total image digits counter e.g., img00001.png where digits = 5
	parameter7:  read gray or color image flag
	*/
	vid = Matrixu::LoadVideo((dataDir+"imgs/").c_str(),"data1", "bmp", (int)frameb(0), (int)frameb(1), 4, false);
	SimpleTracker tr;
	SimpleTrackerParams		trparams;
	BoostingTrackerParams	btparams;
	vector<Matrixu>			saveseq;
	string paramname = "";
	

	////////////////////////////////////////////////////////////////////////////////////////////
	// PARAMETERS	
	Rect initBB = Rect((int)initstate(0,1), (int)initstate(0,0), (int)initstate(0,3), (int)initstate(0,2));

	ClfStrongParams			*clfparams;
	
	// strong model
	switch( strong ){
		case 1:		// MILTrack
			clfparams = new ClfMilBoostParams();
			((ClfMilBoostParams*)clfparams)->_numSel		= 50;
			((ClfMilBoostParams*)clfparams)->_numFeat		= 250;
			paramname += "_MIL";
			trparams._posradtrain							= 4.0f;
			trparams._negnumtrain							= 65;
			break;
		case 2:		// OBA1
			clfparams = new ClfAdaBoostParams();
			((ClfAdaBoostParams*)clfparams)->_numSel		= 50;
			((ClfAdaBoostParams*)clfparams)->_numFeat		= 250;
			paramname += "_OAB1";
			trparams._posradtrain							= 1.0f;
			trparams._negnumtrain							= 65;
			break;
		case 3:		// OBA5
			clfparams = new ClfAdaBoostParams();
			((ClfAdaBoostParams*)clfparams)->_numSel		= 50;
			((ClfAdaBoostParams*)clfparams)->_numFeat		= 250;
			paramname += "_OAB5";
			trparams._posradtrain							= 4.0f;
			trparams._negnumtrain							= 65;
			break;

		default:
			abortError(__LINE__,__FILE__,"Error: invalid classifier choice.");
	}

	// feature parameters
	FtrParams *ftrparams;
	HaarFtrParams haarparams;
	ftrparams = &haarparams;

	clfparams->_ftrParams	= ftrparams;

	// tracking parameters
	trparams._init_negnumtrain = 65;
	trparams._init_postrainrad = 3.0f;
	trparams._initstate[0]	= initstate(0,0);
	trparams._initstate[1]	= initstate(0,1);
	trparams._initstate[2]	= initstate(0,2);
	trparams._initstate[3]	= initstate(0,3);
	trparams._srchwinsz		= 7;   //25
	trparams._negsamplestrat = 1;
	trparams._initWithFace	= false;

	trparams._debugv		= false;
	trparams._disp			= false; // set this to true if you want to see video output (though it slows things down)
	trparams._vidsave		= savevid? dataDir + name + paramname + "_TR" + int2str(trial,3) + ".avi" : "";
	trparams._trsave		= dataDir + name + paramname + "_TR" + int2str(trial,3) + "_c.txt";

	//Boosting Tracking params

	btparams.numBaseClassifier = 100;
	btparams.overlap = 0.99;
	btparams.searchFactor = 2;
	btparams.initBB = Rect((int)initstate(0,1), (int)initstate(0,0), (int)initstate(0,3), (int)initstate(0,2));

	////////////////////////////////////////////////////////////////////////////////////////////
	// TRACK

	
	cout << "\n===============================================\nTRACKING: " << name + paramname + "_TR" + int2str(trial,3) << endl;
	cout <<   "-----------------------------------------------\n";

	tr.track_frames(vid, trparams, btparams, clfparams);
	cout << endl << endl;
	delete clfparams;

	
}

void BoostingTrack(int numBaseClassifier, float overlap, float searchFactor, Rect initBB, vector<Matrixu> &vid)
{
	Rect trackingRect;
	unsigned char *curFrame=NULL;
	int key;
	int frameNum = 0;
	Matrixu frameDisp;
	//choose the image source
	//ImageSource *imageSequenceSource;
	//imageSequenceSource = new ImageSourceDir(source);
	//ImageHandler* imageSequence = new ImageHandler (imageSequenceSource);
	//imageSequence->getImage();

	//imageSequence->viewImage ("Tracking...", false);
	//cvSetMouseCallback( "Tracking...", on_mouse, 0 );

	printf("1.) Select bounding box to initialize tracker.\n");
	printf("2.) Switch to DOS-window and press enter when done.\n");

	trackingRect = initBB;

	IplImage *srcImg = vid[frameNum].getIpl();
	IplImage *dstImg = cvCreateImage( cvGetSize(srcImg), srcImg->depth, 1 );
	cvCvtColor( srcImg, dstImg, CV_BGR2GRAY );
	curFrame = reinterpret_cast<unsigned char*>(dstImg->imageData);
	frameNum++;
	Size frameSize;
	frameSize.height = vid[0].rows();
	frameSize.width = vid[0].cols();

	ImageRepresentation* curFrameRep = new ImageRepresentation(curFrame, frameSize); 
	Rect wholeImage;
	wholeImage = frameSize;

	printf ("init tracker...");
	BoostingTracker* tracker;
	tracker = new BoostingTracker (curFrameRep, trackingRect, wholeImage, numBaseClassifier);
	printf (" done.\n");

	Size trackingRectSize;
	trackingRectSize = trackingRect;
	printf ("start tracking (stop by pressing any key)...\n\n");

	//FILE* resultStream;
	//if (resultDir[0]!=0)
	//{
	//	char *myBuff = new char[255];
	//	sprintf (myBuff, "%s/BoostingTracker.txt", resultDir);
	//	resultStream = fopen(myBuff, "w");
	//	delete[] myBuff;
	//}

	int counter= 0;
	bool trackerLost = false;

	key=(char)-1;
	//tracking loop
	while (key==(char)-1)
	{
		clock_t timeWatch;
		timeWatch = clock();

		//do tracking
		counter++;

		frameDisp = vid[frameNum];
		//curFrame = (unsigned char*)vid[frameNum].getIpl()->imageData;
		srcImg = vid[frameNum].getIpl();
		dstImg = cvCreateImage( cvGetSize(srcImg), srcImg->depth, 1 );
		cvCvtColor( srcImg, dstImg, CV_BGR2GRAY );
		curFrame = reinterpret_cast<unsigned char*>(dstImg->imageData);
		frameNum++;
		/*if (curFrame!=NULL)
			delete[] curFrame;

		curFrame = imageSequence->getGrayImage ();
		if (curFrame == NULL)
			break;*/

		//calculate the patches within the search region
		Patches *trackingPatches;	
		Rect searchRegion;
		searchRegion = tracker->getTrackingROI(searchFactor);
		trackingPatches = new PatchesRegularScan(searchRegion, wholeImage, trackingRectSize, overlap);

		curFrameRep->setNewImageAndROI(curFrame, searchRegion);

		if (!tracker->track(curFrameRep, trackingPatches))
		{
			trackerLost = true;
			break;
		}

		delete trackingPatches;

		//display results
		Rect trackedRect = tracker->getTrackedPatch();
		frameDisp.drawRect(trackedRect.width, trackedRect.height, trackedRect.left, trackedRect.upper, 1, 0,
			3, 255, 0, 0 );
		frameDisp.display(1);
		cvWaitKey(1);

		//write images
		//if (resultDir[0]!=0)
		//{
		//	if (trackerLost)
		//		fprintf (resultStream, "%8d 0 0 0 0 -1\n", counter);
		//	else
		//		fprintf (resultStream, "%8d %3d %3d %3d %3d %5.3f\n", counter, tracker->getTrackedPatch().left, tracker->getTrackedPatch().upper, tracker->getTrackedPatch().width, tracker->getTrackedPatch().height, tracker->getConfidence());

		//	char *myBuff = new char[255];
		//	sprintf (myBuff, "%s/frame%08d.jpg", resultDir, counter+2);
		//	imageSequence->saveImage(myBuff);
		//}

		// wait for opencv display
		key=cvWaitKey(25);

//#if OS_type==2
//		if (kbhit())
//			key=(char)0;
//#endif
		timeWatch=clock()-timeWatch;
		double framesPerSecond=1000.0/timeWatch;
		printf("TRACKING: confidence: %5.3f  fps: %5.2f   \r", tracker->getConfidence(), framesPerSecond);


	}

	printf ("\n\ntracking stopped\n");
	if (trackerLost) 
	{
		printf ("tracker lost\n");
//#if OS_type==2
//		getch();
//#endif
#if OS_type==1
		std::cin.get();
#endif


	}


	//if (resultDir[0]!=0)
	//	fclose(resultStream);

	//clean up
	delete tracker;
	//delete imageSequenceSource;
	//delete imageSequence;
	if (curFrame == NULL)
		delete[] curFrame;
	delete curFrameRep;


}

void	FaceTrackDemo(const char* savefile)
{
	float vwidth = 240, vheight = 180;  // images coming from webcam will be resized to these dimensions (smaller images = faster runtime)
	ClfStrongParams			*clfparams;
	SimpleTracker			tr;
	SimpleTrackerParams		trparams;

	////////////////////////////////////////////////////////////////////////////////////////////
	// PARAMS

	// strong model
	switch( 2 ){
		case 1:		// OBA1
			clfparams = new ClfAdaBoostParams();
			((ClfAdaBoostParams*)clfparams)->_numSel	= 50;
			((ClfAdaBoostParams*)clfparams)->_numFeat	= 250;
			trparams._posradtrain						= 1.0f;
			break;
		case 2:		// MILTrack
			clfparams = new ClfMilBoostParams();
			((ClfMilBoostParams*)clfparams)->_numSel	= 50;
			((ClfMilBoostParams*)clfparams)->_numFeat	= 250;
			trparams._posradtrain						= 4.0f;
			break;
		
	}

	// feature parameters
	FtrParams *ftrparams;
	HaarFtrParams haarparams;
	ftrparams = &haarparams;
	clfparams->_ftrParams	= ftrparams;

	// online boosting parameters
	clfparams->_ftrParams		= ftrparams;

	// tracking parameters
	trparams._init_negnumtrain	= 65;
	trparams._init_postrainrad	= 3.0f;
	trparams._srchwinsz			= 25;
	trparams._negnumtrain		= 65;


	// set up video
	CvCapture* capture = cvCaptureFromCAM( 0 );
	if( capture == NULL ){
		abortError(__LINE__,__FILE__,"Camera not found!");
		return;
	}


	////////////////////////////////////////////////////////////////////////////////////////////
	// TRACK

	// print usage
	fprintf(stderr,"MILTRACK FACE DEMO\n===============================\n");
	fprintf(stderr,"This demo uses the OpenCV face detector to initialize the tracker.\n");
	fprintf(stderr,"Commands:\n");
	fprintf(stderr,"\tPress 'q' to QUIT\n");
	fprintf(stderr,"\tPress 'r' to RE-INITIALIZE\n\n");
	
	// grab first frame
	Matrixu frame,framer,framedisp;
	Matrixu::CaptureImage(capture,framer);
	frame = framer.imResize(vheight,vwidth);

	// save output
	CvVideoWriter *w=NULL;

	// initialize with face location
	while( !Tracker::initFace(&trparams,frame) ){
		Matrixu::CaptureImage(capture,framer);
		frame = framer.imResize(vheight,vwidth);
		frame.display(1,2);
	}
	ftrparams->_height		= (uint)trparams._initstate[2];
	ftrparams->_width		= (uint)trparams._initstate[3];
	tr.init(frame, trparams, clfparams);

	StopWatch sw(true);
	double ttime=0.0;
	double probtrack=0.0;

	// track
	for (int cnt = 0; Matrixu::CaptureImage(capture,framer); cnt++) {
		frame = framer.imResize(vheight,vwidth); 
		tr.track_frame(frame, framedisp);  // grab tracker confidence

		// initialize video output
		if( savefile != NULL && w==NULL ){
			w = cvCreateVideoWriter( savefile, CV_FOURCC('I','Y','U','V'), 15, cvSize(framedisp.cols(), framedisp.rows()), 3 );
		}

		// display and save frame
		framedisp._keepIpl=true;
		framedisp.display(1,2);
		if( w != NULL )
			Matrixu::WriteFrame(w, framedisp);
		framedisp._keepIpl=false; framedisp.freeIpl();
		char q=cvWaitKey(1);
		ttime = sw.Elapsed(true);
		fprintf(stderr,"%s%d Frames/%f sec = %f FPS, prob=%f",ERASELINE,cnt,ttime,((double)cnt)/ttime,probtrack);
	
		// quit
		if( q == 'q' )
			break;

		// restart with face detector
		else if( q == 'r' || probtrack<0 ) 
		{
			while( !Tracker::initFace(&trparams,frame) && q!='q' ){
				Matrixu::CaptureImage(capture,framer);
				frame = framer.imResize(vheight,vwidth);
				frame.display(1,2);
				q=cvWaitKey(1);
			}
			if( q == 'q' )
				break;
			
			// re-initialize tracker with new parameters
			ftrparams->_height		= (uint)trparams._initstate[2];
			ftrparams->_width		= (uint)trparams._initstate[3];
			clfparams->_ftrParams	= ftrparams;
			fprintf(stderr,"\n");
			tr.init(frame, trparams, clfparams);
			
		}
	}


	// clean up
	if( w != NULL )
		cvReleaseVideoWriter( &w );
	cvReleaseCapture( &capture );

}

