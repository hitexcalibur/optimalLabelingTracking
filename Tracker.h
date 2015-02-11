// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License 
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

#ifndef TRACKER_PUBLIC
#define TRACKER_PUBLIC

#include "OnlineBoost.h"
#include "Public.h"

#include ".\Online Boosting\Boostiong Tracker\BoostingTracker.h"
#include ".\Online Boosting\Semi Boosting Tracker\SemiBoostingTracker.h"
#include ".\Online Boosting\Beyond Semi Boosting Tracker\BeyondSemiBoostingTracker.h"
#include ".\OnlineBoosting\Regions.h"

#include ".\gsl\data.h"
#include ".\gsl\prob_functions.h"

#include ".\Fragtrack\Fragments_Tracker.h"

#include ".\Surf\Particle.h"
#include ".\Surf\surf.h"

#define DIS_THRESHOLD 5
#define TRACKERADJUST 0.05


//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

class TrackerParams
{
public:
					TrackerParams();

	vectori			_boxcolor;						// for outputting video
	uint			_lineWidth;						// line width 
	uint			_negnumtrain,_init_negnumtrain; // # negative samples to use during training, and init
	float			_posradtrain,_init_postrainrad; // radius for gathering positive instances
	uint			_posmaxtrain;					// max # of pos to train with
	bool			_debugv;						// displays response map during tracking [kinda slow, but help in debugging]
	vectorf			_initstate;						// [x,y,scale,orientation] - note, scale and orientation currently not used
	bool			_useLogR;						// use log ratio instead of probabilities (tends to work much better)
	bool			_initWithFace;					// initialize with the OpenCV tracker rather than _initstate
	bool			_disp;							// display video with tracker state (colored box)

	string			_vidsave;						// filename - save video with tracking box
	string			_trsave;						// filename - save file containing the coordinates of the box (txt file with [x y width height] per row)

};



class SimpleTrackerParams : public TrackerParams
{
public:
					SimpleTrackerParams();

	uint			_srchwinsz;						// size of search window
	uint			_negsamplestrat;				// [0] all over image [1 - default] close to the search window
};

class BoostingTrackerParams : public TrackerParams
{
public:
	BoostingTrackerParams();

	int numBaseClassifier;
	float overlap, searchFactor;
	Rect initBB;

};

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

class Tracker
{
public:
	
	static bool		initFace(TrackerParams* params, Matrixu &frame);
	static void		replayTracker(vector<Matrixu> &vid, string states, string outputvid="",uint R=255, uint G=0, uint B=0);
	static void		replayTrackers(vector<Matrixu> &vid, vector<string> statesfile, string outputvid, Matrixu colors);

protected:
	static CvHaarClassifierCascade	*facecascade;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

class SimpleTracker : public Tracker
{
public:

					SimpleTracker();
					~SimpleTracker();
	double			track_frame(Matrixu &frame, Matrixu &framedisp, uchar* curBackground); // track object in a frame;  requires init() to have been called.
	double			track_frame(Matrixu &frame, Matrixu &framedisp); // track object in a frame;  requires init() to have been called.
	void			track_frames(vector<Matrixu> &video, SimpleTrackerParams p, BoostingTrackerParams btparams, ClfStrongParams *clfparams);  // initializes tracker and runs on all frames
	bool			init(Matrixu frame, SimpleTrackerParams p, ClfStrongParams *clfparams);
	Matrixf &		getFtrHist() { return _clf->_ftrHist; }; // only works if _clf->_storeFtrHistory is set to true.. mostly for debugging

	void		backgroundModel(unsigned char* curFrameGray, unsigned char* BGMGray, int cols, int rows, Rect trackedRect);  //for beyond semi tracker's bgm update

	int			calculateTrackingResult(int imageNum, Patches* patches);
	Rect			getTrackingResult(){return m_trackedResult;};
	void			EM(Dataset *data);

	int*			calculateLabels(float* confidences, int imageNum);

	void			prepareLabels(int imageNum);

	void			calculatePrior();

	int			calculateTrackingResult_versionB(Patches* trackingPatches, SampleSet sampleSet);

	bool            inTopTen(int idx, float* confidence, int imageNum);

	void			validJudge();


private:
	ClfStrong			*_clf;
	vectorf				_curState;
	SimpleTrackerParams	_trparams;
	ClfStrongParams		*_clfparams;
	int					_cnt;

	float*				MILconfidences;
	int					MILmaxIdx;

	int*				m_MILlabels;
	int*				m_BoostingTrackinglabels;
	int*				m_SemiBoostingTrackinglabels;
	int*				m_BeyongdSemiBoostingTrackinglabels;

	int**				m_labels;

	bool*				trackedValid;

	float*				priorTracker;

	int					trackerNum;
	float					validJudgeThershold;

	BoostingTracker*		_boostingTracker;
	BoostingTrackerParams _btparams;

	SemiBoostingTracker*	_semiBoostingTracker;
	BeyondSemiBoostingTracker* _beyondSemiBoostingTracker;

	Fragments_Tracker*		_fragTracker;
	Parameters				_fragParameters;

	CParticleFilterTrack	_surfTracker;



	Rect					m_trackedResult;
	bool				m_wholetrackedLost;

	int						frameind;

	ofstream			combineOut;
	ofstream			MILout;
	ofstream			OnlineOut;
	ofstream			SemiOut;
	ofstream			BeyondOut;
	ofstream			SurfOut;
	ofstream			FragOut;

	ofstream			trackerCondidencOut;

	
};



#endif



