/**
 * Initiates the BRIEF loop detector
 * Inputs:
 * 	@VOC_FILE : is the file containing the vocabulary to use
 *  Example : ./dloopdetector_Brief path/
 * Uses FAST for feature extraction and BRIEF for feature description
 */

#include <iostream>
#include <vector>
#include <string>

// DLoopDetector and DBoW2
#include "DLoopDetector/DBoW2.h" // defines BriefVocabulary
#include "DLoopDetector/DLoopDetector.h" // defines BriefLoopDetector
#include "DLoopDetector/DVision.h" // Brief 

// OpenCV
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "LoopDetector.h"

using namespace DLoopDetector;
using namespace DBoW2;
using namespace DVision;
using namespace std;

// ----------------------------------------------------------------------------

static const char *VOC_FILE = "../../resources/brief_k10L6.voc.gz";
static const char *IMAGE_DIR = "../../resources/images_1Hz/round_1";
static const char *POSE_FILE = "/home/giko/oh-distro-private/software/place_recognition_distro/dloopdetector_lcm/resources/husky_trajectory/NEW_TRAJ.txt";

static const int IMAGE_W = 1024; // image size
static const int IMAGE_H = 1024;
static const char *BRIEF_PATTERN_FILE = "/home/giko/oh-distro-private/software/place_recognition_distro/dloopdetector_lcm/resources/brief_pattern.yml";

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

/// This functor extracts BRIEF descriptors in the required format
class BriefExtractor: public FeatureExtractor<FBrief::TDescriptor>
{
public:
  /** 
   * Extracts features from an image
   * @param im image
   * @param keys keypoints extracted
   * @param descriptors descriptors extracted
   */
  virtual void operator()(const cv::Mat &im, 
    vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const;

  /**
   * Creates the brief extractor with the given pattern file
   * @param pattern_file
   */
  BriefExtractor(const std::string &pattern_file);

private:

  /// BRIEF descriptor extractor
  DVision::BRIEF m_brief;
};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

int main(int argc, char** argv)
{
  float alpha = 0.3;
  int k = 1;
  if ( argc>1 )
  {
    VOC_FILE = argv[1];
    cout<<VOC_FILE<<endl;
    IMAGE_DIR = argv[2];
  }
  
  if (argc>3)// arc > 5
  {
    alpha = atof(argv[3]);
    k = atoi(argv[4]);
    cout<<"Alpha = "<<alpha<<endl;
    cout<<"K = "<<k<<endl;
  }
  // prepares the demo
  LoopDetector<BriefVocabulary, BriefLoopDetector, FBrief::TDescriptor> 
    demo(VOC_FILE, IMAGE_DIR, POSE_FILE, IMAGE_W, IMAGE_H);
  
  try 
  {
    // run the demo with the given functor to extract features
    // default value of brief descriptor is 32 byte
    BriefExtractor extractor(BRIEF_PATTERN_FILE);
    demo.run("BRIEF", extractor, alpha, k);
  }
  catch(const std::string &ex)
  {
    cout << "Error: " << ex << endl;
  }

  return 0;
}

// ----------------------------------------------------------------------------

BriefExtractor::BriefExtractor(const std::string &pattern_file)
{
  // The DVision::BRIEF extractor computes a random pattern by default when
  // the object is created.
  // We load the pattern that we used to build the vocabulary, to make
  // the descriptors compatible with the predefined vocabulary
  
  // loads the pattern
  cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
  if(!fs.isOpened()) throw string("Could not open file ") + pattern_file;
  
  vector<int> x1, y1, x2, y2;
  fs["x1"] >> x1;
  fs["x2"] >> x2;
  fs["y1"] >> y1;
  fs["y2"] >> y2;
  
  m_brief.importPairs(x1, y1, x2, y2);
  // DVision::BRIEF m(512,48,DVision::BRIEF::RANDOM_CLOSE);
  // m_brief = m;
}

// ----------------------------------------------------------------------------

void BriefExtractor::operator() (const cv::Mat &im, 
  vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const
{
  // extract FAST keypoints with opencv
  const int fast_th = 20; // corner detector response threshold
  cv::FAST(im, keys, fast_th, true);
  
  // compute their BRIEF descriptor
  m_brief.compute(im, keys, descriptors,false);
}

// ----------------------------------------------------------------------------

