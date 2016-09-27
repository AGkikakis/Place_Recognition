/* Tool for creating a vocabulary, using DBoW2 from a set of frames
 * 
 * Generates generate_vocabulary excecutable
 * Input:
 *    	@ the path containing the images to generate the vocabulary
 *      @ branching_factor
 *      @ maximum depth of the tree
 *      @ boolean denoting use of extended surf
 *	example: ./generate_vocabulary path/ 10 6 1
 * Output:
 *      @ a vocabulary .txt file named after its variables
 *	  e.g 'vocabulary_husky_all_k=10_L=6_ext.txt' or
 *	 'vocabulary_husky_all_k=10_L=6.txt' if extended_surf=0
 */

#include <iostream>
#include <vector>
#include <dirent.h>
#include <string>
#include <algorithm>
#include <sys/stat.h>
#include <unistd.h>
#include <typeinfo>

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

// Generic detector, initializes and sets the variables for the loop detection
#include "LoopDetector.h"
#include "DLoopDetector/FBrief.h"

using namespace DLoopDetector;
using namespace DBoW2;
using namespace DVision;
using namespace std;

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


void countImages(int &num,vector<string> &fileNames);
void loadFeatures(vector< vector< FBrief::TDescriptor> > &descriptors);
void createVocabulary(const vector< vector< FBrief::TDescriptor> > &descriptors);

// number of training images
int Nimages;
// names of images
vector<string> fileNames;

//default directory containing training images
string dir_name = "husky_images_per_round/round_1/";

static const char *BRIEF_PATTERN_FILE = "../../resources/brief_pattern.yml";

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

// branching factor and depth levels
// L = maximum depth of tree  | k = maximum children
int k = 3;
int L = 3;

int main(int argc, char** argv)
{
  if ( argc>1 and argc<=5)
  {
    dir_name = argv[1];
    k=atoi(argv[2]);
    L=atoi(argv[3]);
  }
  cout<<"k = "<<k<<" | L = "<<L<<endl;
  
  vector< vector< FBrief::TDescriptor > > features;
  //count number of images and get their names
  countImages(Nimages,fileNames);
  //Extract all features from all images
  vector<FBrief::TDescriptor> descriptors;
  loadFeatures(features);
  //Create the vocabulary and save it to a txt file
  createVocabulary(features);
  //createVocabulary(descriptors);
  return 0;
}


// locates the keypoints and their descriptors
// returns a vector containing for each image the 128byte surf descriptor vector
void loadFeatures(vector< vector< FBrief::TDescriptor > > &descriptors)
{
  descriptors.clear();
  descriptors.reserve(Nimages);
  BriefExtractor extractor(BRIEF_PATTERN_FILE);
  
  cout << "Extracting ";
  cout<<"BRIEF features..." << endl;
  for(int i = 0; i < Nimages; i++)
  {
    stringstream ss;
    //save to ss the name of the image
    ss << dir_name <<fileNames[i];
    cout<<"image is: "<<ss.str()<<endl;
    //load image
    cv::Mat image = cv::imread(ss.str(), -1);//read image as is
    cv::Mat mask;

    vector<cv::KeyPoint> keypoints;
    //vector that contains all descriptors together
    //FBrief::TDescriptor allImagesFeatures;
    vector<FBrief::TDescriptor> v;

    //loads all features(descriptors) , keypoints and the mask used to extract keypoints
    extractor(image,keypoints, v);
    //rearange to the form : image<descriptors<descriptor(64 or 128 byte)> > >
    descriptors.push_back(v);
  }
}


void createVocabulary(const vector< vector< FBrief::TDescriptor> > &descriptors)
{
  TemplatedVocabulary< DBoW2::FBrief::TDescriptor, DBoW2::FBrief > voc(k,L,TF_IDF,L1_NORM);
  cout<<voc.getDepthLevels()<<endl;//create(features);
  cout<<voc.getBranchingFactor()<<endl;
  //voc.setScoringType(L1_NORM);

  // // creating hierarhical tree of k^L nodes
  cout << "Creating a " << k << "^" << L << " vocabulary..." << endl;
  voc.create(descriptors);
  cout << "... done!" << endl;
  //
  cout << "Stopping some words..." << endl;
  // stop 1% of words
  cout<<"Number of words stoped : "<<voc.stopWords(1)<<endl;
  
  cout << "Vocabulary information: " << endl;
  cout<<"Weighting type : "<< voc.getWeightingType()<<endl;
  cout<<"Scoring type : "<<voc.getScoringType()<<endl;
  cout<<"Number of words : "<<voc.size()<<endl;
  //cout<<"Number of words stoped : "<<stopw<<endl;

  stringstream st;
  cout << endl << "Saving vocabulary..." << endl;
  st<<"vocabulary_husky_all_BRIEF_k="<<k<<"_L="<<L<<".txt";
  voc.save(st.str().c_str());
  cout << "Done" << endl;

}


bool has_suffix(const string& s, const string& suffix)
{
    return (s.size() >= suffix.size()) && equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

//function that counts the number of frames in the given folder
void countImages(int &num, vector<string> &fileNames)
{
  DIR *dir= opendir(dir_name.c_str());
  num=0;
  if(!dir)
      {
          cout<<"Could not find directory: "<<dir_name.c_str()<<endl;
      }

  dirent *entry;
  while ((entry = readdir (dir)) != NULL)
  {
            if(has_suffix(entry->d_name, ".png"))
          {
                num++;
      fileNames.push_back(entry->d_name);
          }
      }
      closedir(dir);
  cout<<"Found "<<num<<" png files"<<endl;
}

inline bool file_exists(const string& name)
{
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
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
  cout<<"Desc length = "<< m_brief.getDescriptorLengthInBits()<<endl;
  // compute their BRIEF descriptor
  m_brief.compute(im, keys, descriptors,false);
}

// ----------------------------------------------------------------------------


