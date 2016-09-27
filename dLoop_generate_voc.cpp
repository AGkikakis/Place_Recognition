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
#include <ctime>

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

using namespace DLoopDetector;
using namespace DBoW2;
using namespace std;

void countImages(int &num,vector<string> &fileNames);
void loadFeatures(vector< vector< FSurf64::TDescriptor> > &descriptors);
void createVocabulary(const vector< vector< FSurf64::TDescriptor> > &descriptors);
void resize_descriptors(FSurf64::TDescriptor &allImagesFeatures,vector<FSurf64::TDescriptor> &v);

// number of training images
int Nimages;
// names of images
vector<string> fileNames;
// extended surf gives 128-dimensional vectors else 64-d
bool extended_surf = true;

//default directory containing training images
string dir_name = "husky_images_per_round/round_1/";

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
    extended_surf = (atoi(argv[4]) != 0 );
  }

  if (extended_surf)
    FSurf64::setDimensions(128);
  else
    FSurf64::setDimensions(64);

  vector< vector< FSurf64::TDescriptor > > features;
  //count number of images and get their names
  countImages(Nimages,fileNames);
  //Extract all features from all images
  vector<FSurf64::TDescriptor> descriptors;
  loadFeatures(features);
  //Create the vocabulary and save it to a txt file
  createVocabulary(features);
  //createVocabulary(descriptors);
  return 0;
}


// locates the keypoints and their descriptors
// returns a vector containing for each image the 128byte surf descriptor vector
void loadFeatures(vector< vector< FSurf64::TDescriptor > > &descriptors)
{
  descriptors.clear();
  descriptors.reserve(Nimages);
  // hessian threshold=400
  cv::SURF surf(400,4,2,extended_surf,0);

  cout << "Extracting ";
  if (extended_surf){cout<<"extended ";}
  cout<<"SURF features..." << endl;
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
    FSurf64::TDescriptor allImagesFeatures;
    vector<FSurf64::TDescriptor> v;
    clock_t start, stop;
    double totalTime;

    start = clock();
    //loads all features(descriptors) , keypoints and the mask used to extract keypoints
    surf(image, mask,keypoints, allImagesFeatures);stop = clock();
    totalTime = (stop - start) / (double)CLOCKS_PER_SEC;
    cout<<"Ellapsed time = "<<totalTime<<endl;
    //rearange to the form : image<descriptors<descriptor(64 or 128 byte)> > >
    resize_descriptors(allImagesFeatures,v);
    descriptors.push_back(v);
  }
}

void resize_descriptors(FSurf64::TDescriptor &allImagesFeatures,vector<FSurf64::TDescriptor> &v)
{
  v.reserve(v.size()+allImagesFeatures.size()/(extended_surf ? 128 : 64));
  typename FSurf64::TDescriptor::const_iterator it;
  typename FSurf64::TDescriptor::const_iterator itt;
  int step = (extended_surf ? 128 : 64);

  for (it = allImagesFeatures.begin(); it != allImagesFeatures.end(); it=it+step)
  {
    FSurf64::TDescriptor descriptor;
    descriptor.reserve(extended_surf ? 128 : 64 );
    for(itt=it; itt!=it+step; ++itt)
    {
      descriptor.push_back(*itt);
    }
    v.push_back(descriptor);
  }
}

void createVocabulary(const vector< vector< FSurf64::TDescriptor> > &descriptors)
{
  TemplatedVocabulary< DBoW2::FSurf64::TDescriptor, DBoW2::FSurf64 > voc(k,L,TF_IDF,L1_NORM);
  cout<<voc.getDepthLevels()<<endl;//create(features);
  cout<<voc.getBranchingFactor()<<endl;
  //voc.setScoringType(L1_NORM);

  // creating hierarhical tree of k^L nodes
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

  stringstream st;
  cout << endl << "Saving vocabulary..." << endl;
  if (extended_surf){st<<"vocabulary_husky_all_k="<<k<<"_L="<<L<<"_ext.txt";}
  else{st<<"vocabulary_husky_all_k="<<k<<"_L="<<L<<".txt";}
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
