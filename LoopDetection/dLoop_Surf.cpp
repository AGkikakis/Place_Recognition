/**
 * Initiates the SURF loop detector
 * Inputs:
 * 	@VOC_FILE : is the file containing the vocabulary to use
 * 	@extended_surf : denotes the use of 128 or 64 bit descriptors
 *  Example : ./dloopdetector_Surf path/ 1
 */

#include <iostream>
#include <vector>
#include <string>

// DLoopDetector and DBoW2
#include "DLoopDetector/DBoW2.h" // defines Surf64Vocabulary
#include "DLoopDetector/DLoopDetector.h" // defines Surf64LoopDetector
#include "DLoopDetector/DUtilsCV.h" // defines macros CVXX

// OpenCV
#include <opencv/cv.h>
#include <opencv/highgui.h>
#if CV24
#include <opencv2/nonfree/features2d.hpp>
#endif

// Demo
#include "LoopDetector.h"

using namespace DLoopDetector;
using namespace DBoW2;
using namespace std;

// ----------------------------------------------------------------------------

static const char *VOC_FILE = "../../resources/vocabulary_husky_all_k=10_L=4.txt.gz";
static const char *IMAGE_DIR = "../../resources/images_1Hz/test_images";
//static const char *POSE_FILE = "./resources/pose.txt";
static const char *POSE_FILE = "/home/giko/oh-distro-private/software/place_recognition_distro/dloopdetector_lcm/resources/husky_trajectory/NEW_TRAJ.txt";
static const int IMAGE_W = 1024; // image size
static const int IMAGE_H = 1024;
bool extended_surf = false;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/// This functor extracts SURF64 descriptors in the required format
class SurfExtractor: public FeatureExtractor<FSurf64::TDescriptor>
{
public:
  /**
   * Extracts features from an image
   * @param im image
   * @param keys keypoints extracted
   * @param descriptors descriptors extracted
   */
  virtual void operator()(const cv::Mat &im,
    vector<cv::KeyPoint> &keys, vector<vector<float> > &descriptors) const;
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
    extended_surf = (atoi(argv[3]) != 0 );
  }
  
  if (argc>4)// arc > 5
  {
    alpha = atof(argv[4]);
    k = atoi(argv[5]);
    cout<<"Alpha = "<<alpha<<endl;
    cout<<"K = "<<k<<endl;
  }

  if (extended_surf)
    FSurf64::setDimensions(128);
  else
    FSurf64::setDimensions(64);
  // prepares the detector
  // given the vocabulary files, method starts adding frames and checking for
  // loops in each new frame added
  LoopDetector<Surf64Vocabulary, Surf64LoopDetector, FSurf64::TDescriptor>
    demo(VOC_FILE, IMAGE_DIR, POSE_FILE, IMAGE_W, IMAGE_H);

  try
  {
    // run the demo with the given functor to extract features
    SurfExtractor extractor;
    if ( extended_surf )
      demo.run("SURF128", extractor, alpha, k);
    else
      demo.run("SURF64", extractor, alpha, k);
  }
  catch(const std::string &ex)
  {
    cout << "Error: " << ex << endl;
  }

  return 0;
}

// ----------------------------------------------------------------------------

void SurfExtractor::operator() (const cv::Mat &im,
  vector<cv::KeyPoint> &keys, vector<vector<float> > &descriptors) const
{
  // extract surfs with opencv
  static cv::SURF surf_detector(400,4,2,extended_surf,0);

  keys.clear(); // opencv 2.4 does not clear the vector
  vector<float> plain;
  surf_detector(im, cv::Mat(), keys, plain);

  // change descriptor format
  const int L = surf_detector.descriptorSize();
  descriptors.resize(plain.size() / L);

  unsigned int j = 0;
  for(unsigned int i = 0; i < plain.size(); i += L, ++j)
  {
    descriptors[j].resize(L);
    std::copy(plain.begin() + i, plain.begin() + i + L, descriptors[j].begin());
  }
}

// ----------------------------------------------------------------------------
