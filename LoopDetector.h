/**
 * File for Template Vocabulary, Detector and Desciptor
 * Inputs:
 *	:@ a vocabulary file
 * 	:@ path containing a set of frames
 *	:@ a pose file
 *	:@ widht of the frames
 *	:@ height of the frames
 * Exctaracts the features from the frames sequentially and searches
 * for loops in each new given frame
 */

#ifndef __LOOP_DETECTOR__
#define __LOOP_DETECTOR__

#include <iostream>
#include <vector>
#include <string>

// OpenCV
#include <opencv/cv.h>
#include <opencv/highgui.h>

// DLoopDetector and DBoW2
#include "DLoopDetector/DBoW2.h"
#include "DLoopDetector/DLoopDetector.h"
#include "DLoopDetector/DUtils.h"
#include "DLoopDetector/DUtilsCV.h"
#include "DLoopDetector/DVision.h"
#include "Evaluation.h"

using namespace DLoopDetector;
using namespace DBoW2;
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/// Generic class to create functors to extract features
template<class TDescriptor>
class FeatureExtractor
{
public:
  /**
   * Extracts features
   * @param im image
   * @param keys keypoints extracted
   * @param descriptors descriptors extracted
   */
  virtual void operator()(const cv::Mat &im,
    vector<cv::KeyPoint> &keys, vector<TDescriptor> &descriptors) const = 0;
};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/// @param TVocabulary vocabulary class (e.g: Surf64Vocabulary)
/// @param TDetector detector class (e.g: Surf64LoopDetector)
/// @param TDescriptor descriptor class (e.g: vector<float> for SURF)
template<class TVocabulary, class TDetector, class TDescriptor>



/// Class to run the demo
class LoopDetector
{
public:

  /**
   * @param vocfile vocabulary file to load
   * @param imagedir directory to read images from
   * @param posefile pose file
   * @param width image width
   * @param height image height
   */
  LoopDetector(const std::string &vocfile, const std::string &imagedir,
    const std::string &posefile, int width, int height);

  ~LoopDetector(){}

  /**
   * Runs the detector
   * @param name detector name
   * @param extractor functor to extract features
   */
  void run(const std::string &name,
    const FeatureExtractor<TDescriptor> &extractor, const float &alpha, const float &k);

protected:

  /**
   * Reads the robot poses from a file
   * @param filename file
   * @param xs
   * @param ys
   */
  void readPoseFile(const char *filename, std::vector<double> &xs,
    std::vector<double> &ys,std::vector<string> &tss) const;

protected:

  std::string m_vocfile;
  std::string m_imagedir;
  std::string m_dbfile;
  std::string m_posefile;
  int m_width;
  int m_height;
};

// ---------------------------------------------------------------------------

template<class TVocabulary, class TDetector, class TDescriptor>
LoopDetector<TVocabulary, TDetector, TDescriptor>::LoopDetector
  (const std::string &vocfile, const std::string &imagedir,
  const std::string &posefile, int width, int height)
  : m_vocfile(vocfile), m_imagedir(imagedir), m_posefile(posefile),
    m_width(width), m_height(height)
{
}

// ---------------------------------------------------------------------------

// returns a subvector containing the (100 - percent)% of the input vector

void getSubVector(const vector<string> &vec, vector<string> &sub_vector, const int &percent,
  const bool &cut_from_the_start)
{
  vector<string>::const_iterator first;
  vector<string>::const_iterator last;
  if (cut_from_the_start)//get percent% of the values starting from index (100-percent)%
  {
    first = vec.begin() + ( (int) ( (float)(vec.size()*percent) / 100.0));
    last = vec.end();
  }
  else // get percent% of the values starting from (percent)%) 
  {

    first = vec.begin();
    last = vec.end() -  ((int) ( (float)(vec.size()*percent) / 100.0));
  }
  vector<string> newVec(first, last);
  sub_vector = newVec;
}

// ---------------------------------------------------------------------------

inline static bool cmp( const string &s1, const string &s2)
{
  return ( strtoll( (s1.substr(0,s2.size())).c_str(), NULL, 10) < strtoll(s2.c_str(), NULL, 10 ) );  
}

// ---------------------------------------------------------------------------

template<class TVocabulary, class TDetector, class TDescriptor>
void LoopDetector<TVocabulary, TDetector, TDescriptor>::run
  (const std::string &name, const FeatureExtractor<TDescriptor> &extractor,
  const float &alpha, const float &k)
{

  // Set loop detector parameters
  typename TDetector::Parameters params(m_height, m_width);
  Evaluation evaluate(4.0f);
  unsigned int dislocal = 290;
  //unsigned int dislocal = 299;
  bool save_database = true;
  //int start_false_negative = 135;
  int start_false_negative = 149;
  // mean number of features detected per frame
  float meanFeat=0.0;
  // distance travelled
  float distance=0.0;
  
  if (name.find("BRIEF") != std::string::npos)
  {
    save_database = false;
    start_false_negative = 434;
  }
  // Parameters given by default are:
  // use nss = true
  // alpha = 0.3
  // k = 3
  // geom checking = GEOM_DI
  // di levels = 0
  string path = "/home/giko/oh-distro-private/software/place_recognition_distro/"
    "dloopdetector_lcm/resources/";
  // We are going to change these values individually:
  params.use_nss = true; // use normalized similarity score instead of raw score
  params.alpha = alpha; // nss threshold, default = 0.3
  params.k = k; // a loop must be consistent with k previous matches, default = 1
  params.geom_check = GEOM_DI; // use direct index for geometrical checking
  params.di_levels = 2; // use two direct index levels

  //params.dislocal = 413; // number of frames to skip
  // skips dislocal + 1 frames
  params.dislocal = dislocal; // number of frames to skip
  // To verify loops you can select one of the next geometrical checkings:
  // GEOM_EXHAUSTIVE: correspondence points are computed by comparing all
  //    the features between the two images.
  // GEOM_FLANN: as above, but the comparisons are done with a Flann structure,
  //    which makes them faster. However, creating the flann structure may
  //    be slow.
  // GEOM_DI: the direct index is used to select correspondence points between
  //    those features whose vocabulary node at a certain level is the same.
  //    The level at which the comparison is done is set by the parameter
  //    di_levels:
  //      di_levels = 0 -> features must belong to the same leaf (word).
  //         This is the fastest configuration and the most restrictive one.
  //      di_levels = l (l < L) -> node at level l starting from the leaves.
  //         The higher l, the slower the geometrical checking, but higher
  //         recall as well.
  //         Here, L stands for the depth levels of the vocabulary tree.
  //      di_levels = L -> the same as the exhaustive technique.
  // GEOM_NONE: no geometrical checking is done.
  //
  // In general, with a 10^6 vocabulary, GEOM_DI with 2 <= di_levels <= 4
  // yields the best results in recall/time.
  // Check the T-RO paper for more information.
  //

  // Load the vocabulary to use
  cout << "Loading " << name << " vocabulary..." << endl;
  stringstream vocab;
  vocab<<path<<m_vocfile;
  //m_vocfile = vocab.str().c_str(); 
  if (access( vocab.str().c_str() , F_OK ) == -1 )
  {
    cout<<"Error! Vocabulary not found!"<<endl;
    return;
  }
  cout<<vocab.str().c_str()<<endl;
  TVocabulary voc(vocab.str().c_str());
  // Initiate loop detector with the vocabulary
  cout << "Processing sequence..." << endl;
  TDetector detector(voc,params);

  stringstream s;
  string token;
  
  token = m_vocfile.substr( m_vocfile.find_last_of("/") + 1 );
  std::istringstream ss(token);

  s<<path<<"databases/database_"<<name<<"_"<<m_vocfile.substr(25);
  m_dbfile = s.str();

  // images to test the algorithm
  stringstream imdir;
  imdir<<path<<m_imagedir;
  vector<string> filenames =
    DUtils::FileFunctions::Dir(imdir.str().c_str(), ".png", true);
    
  stringstream db_path;

  db_path<<path<<"images_1Hz_NEWLOG/round_1_CUT";
  // images to create a database
  vector<string> db_filenames =
    DUtils::FileFunctions::Dir(db_path.str().c_str(), ".png", true);
  vector<string> visited;

  vector<string> all_filenames;
  cout<<m_dbfile<<endl;
  if (access( m_dbfile.c_str(), F_OK ) != -1 )
  {
    cout<<"Loading database from : "<<m_dbfile<<" ..."<<endl;
    //Method 1: Used custom built function to load database
    stringstream key_name,desc_name;
    key_name<<path<<"databases/keypoints_"<<name<<".yml";
    desc_name<<path<<"databases/descriptors_"<<name<<".txt";
    detector.loadDatabase(m_dbfile, key_name.str(), desc_name.str() );
    //------------------------------------------------
    //Method 2: Creates a new database and passes it to the detector
    // TemplatedDatabase<TDescriptor,DBoW2::FSurf64> *m_database=
    //   new TemplatedDatabase<TDescriptor,DBoW2::FSurf64>(m_dbfile);
    // cout<<"SIZE = "<<m_database->size()<<endl;
    // detector.setDatabase(*m_database);
    // cout<<"DB size = "<<detector.getDatabase().size()<<endl;
    // detector.loadFeatures("resources/Keypoints.yml", "resources/descriptors.txt");
    /* Method 3: gives the database to the detector constructor
    TDetector detector(*m_database,params);
    cout<<"Database contains = "<<detector.getDatabase().size()<<" frames"<<endl;
    // detector.clear();
    */
    
    stringstream allF_path;
    allF_path<<path<<"images_1Hz_NEWLOG/all_images";
    all_filenames = DUtils::FileFunctions::Dir(allF_path.str().c_str(), ".png", true);
  }// if no database exists
  else
  {
    all_filenames = filenames;
    //all_filenames = DUtils::FileFunctions::Dir("../../resources/images_1Hz/all", ".png", true);
  }

  // Process images
  vector<cv::KeyPoint> keys;
  vector<TDescriptor> descriptors;

  // load robot poses
  vector<double> xs, ys;
  vector<string> ts;
  // pointer used for accessing pose of specifi timestamp
  long int pointer=1;

  // read all poses from pose file and store them
  readPoseFile(m_posefile.c_str(), xs, ys, ts);

  // we can allocate memory for the expected number of images
  detector.allocate(filenames.size());

  // prepare visualization windows
  DUtilsCV::GUI::tWinHandler win = "Current image";
  DUtilsCV::GUI::tWinHandler win_match = "Matched image";
  DUtilsCV::GUI::tWinHandler winplot = "Trajectory";

  DUtilsCV::Drawing::Plot::Style normal_style(2); // thickness
  DUtilsCV::Drawing::Plot::Style loop_style('r', 2); // color, thickness
  DUtilsCV::Drawing::Plot::Style valid_style('y', 2);
  DUtilsCV::Drawing::Plot::Style false_negative('g', 2);
  DUtilsCV::Drawing::Plot::Style test('b', 2);

  DUtilsCV::Drawing::Plot implot(640, 640,
    - *std::max_element(xs.begin(), xs.end()),
    - *std::min_element(xs.begin(), xs.end()),
    *std::min_element(ys.begin(), ys.end()),
    *std::max_element(ys.begin(), ys.end()), 20);

  // prepare profiler to measure times
  DUtils::Profiler profiler;
  Evaluation::point previous_point;
  int count = 0;
  // go
  for(unsigned int i = 0; i < filenames.size(); ++i)
  {
    // get image
    cv::Mat im = cv::imread(filenames[i], -1); // loads images as is
    // show image
    profiler.profile("features");
    extractor(im, keys, descriptors);
    profiler.stop();
    // add image to the collection and check if there is some loop
    DetectionResult result;
    profiler.profile("detection");
    // use dbow2 to retrieve 50 ranked matches and applies geometrcial and temporal
    // constrains
    detector.detectLoop(keys, descriptors, result);
    if (i==0)
      meanFeat = keys.size();
    else
      meanFeat = meanFeat+((float)(1.0/i))*(keys.size()-meanFeat);;
    profiler.stop();
    if(result.detection())
    {
      //cout << "- Loop found with image " << result.match<< "!"<< endl;
      /* show image for visual verification
      cv::Mat im = cv::imread(filenames[result.match], -1);
      // show image
      cv::imshow("Matched image",im);
      cv::resizeWindow("Matched image",512,512);
      //cv::waitKey(0); // Wait for a keystroke in the window
      //cv::destroyWindow("Matched image"); */
      ++count;
    }
    else
    {
      // cout << "- No loop: ";
      // switch(result.status)
      // {
      //   case CLOSE_MATCHES_ONLY:
      //     cout << "All the images in the database are very recent" << endl;
      //     break;
      // 
      //   case NO_DB_RESULTS:
      //     cout << "There are no matches against the database (few features in"
      //       " the image?)" << endl;
      //     break;
      // 
      //   case LOW_NSS_FACTOR:
      //     cout << "Little overlap between this image and the previous one"
      //       << endl;
      //     break;
      // 
      //   case LOW_SCORES:
      //     cout << "No match reaches the score threshold (alpha: " <<
      //       params.alpha << ")" << endl;
      //     break;
      // 
      //   case NO_GROUPS:
      //     cout << "Not enough close matches to create groups. "
      //       << "Best candidate: " << result.match << endl;
      //     break;
      // 
      //   case NO_TEMPORAL_CONSISTENCY:
      //     cout << "No temporal consistency (k: " << params.k << "). "
      //       << "Best candidate: " << result.match << endl;
      //     break;
      // 
      //   case NO_GEOMETRICAL_CONSISTENCY:
      //     cout << "No geometrical consistency. Best candidate: "
      //       << result.match << endl;
      //     break;
      // 
      //   default:
      //     break;
      // }
    }

    //cout << endl;
    
    if(i > 0)
    {
      // get only the file name without the extension and the path
      // cut extension
      string fname=filenames[i].substr(0, filenames[i].size()-4 );
      // cut path
      fname=fname.substr( fname.find_last_of("/") + 1 );
      // Since timestamps are generated with higher frequency than the frames we used
      // search for the closest timestamp to the current frame timestamp
      while ( ( strstr( ts[pointer].c_str(), fname.c_str()) == NULL ) && (pointer < ts.size()) )
        pointer++;

      if(result.detection())
        implot.line(xs[pointer-1], ys[pointer-1], xs[pointer], ys[pointer], loop_style);
      else
        implot.line(xs[pointer-1], ys[pointer-1], xs[pointer], ys[pointer], normal_style);
      // calculate traveled distance
      if (i==1)
      {
        previous_point.x = xs[pointer];
        previous_point.y = ys[pointer];
      }
      
      Evaluation::point current_point = {xs[pointer], ys[pointer]};
      distance = distance + evaluate.getDistance(current_point, previous_point);
      previous_point.x = xs[pointer];
      previous_point.y = ys[pointer];
      if ( result.detection() )
      {
        // point containing the position of current frame
        Evaluation::point current = {xs[pointer], ys[pointer]};
        
        // find coordinates of matched frame to make the comparison
        //string matched_fname=all_filenames[result.match].substr(0, all_filenames[result.match].size()-4 );
        string matched_fname=all_filenames[result.match].substr(0, all_filenames[result.match].size()-4 );
        matched_fname=matched_fname.substr( matched_fname.find_last_of("/") + 1 );
        // perform binary search on pose file to find matched frame's position
        vector<string>::const_iterator it = std::lower_bound(ts.begin(), ts.end(), matched_fname,cmp);
        int index = it - ts.begin();
        Evaluation::point matched = {xs[index], ys[index]};
        // add matched points
        evaluate.addPoints(current, matched);
        if ( evaluate.isValidMatch(current, matched) )
          implot.line(xs[pointer-1], ys[pointer-1], xs[pointer], ys[pointer], valid_style);
        
      }
      // for BRIEF
      //else if ( i > dislocal )// did not detect
      else
      {
        // for BRIEF
        // if ( i < 299 || i > 427)
        // frame where round 2 reaches previously visited place
        //if( i > 135 )
        //if (i < 135)
        if ( i > start_false_negative )
        {
          evaluate.foundFalseNegative();
          implot.line(xs[pointer-1], ys[pointer-1], xs[pointer], ys[pointer], false_negative);
        }
      }
      
      DUtilsCV::GUI::showImage(implot.getImage(), true, &winplot, 10);
    }
  }// end for

  if(count == 0)
  {
    cout << "No loops found in this image sequence" << endl;
  }
  else
  {
    cout << count << " loops found in this image sequence!" << endl;
  }

  // cout<<"Database size = "<<detector.getDatabase().size()<<endl;
  // cout << endl << "Execution time:" << endl
  //   << " - Feature computation: " << profiler.getMeanTime("features") * 1e3
  //   << " ms/image" << endl
  //   << " - Loop detection: " << profiler.getMeanTime("detection") * 1e3
  //   << " ms/image" << endl;
  //evaluate.setNumberOfFrames(408-2);
  evaluate.setNumberOfFrames(filenames.size()-135);
  evaluate.toString();
  
  if (access( m_dbfile.c_str(), F_OK ) == -1 & save_database)
  {
    cout<<"Saving database.."<<endl;
    stringstream key_name,desc_name;
    key_name<<path<<"databases/keypoints_"<<name<<".yml";
    desc_name<<path<<"databases/descriptors_"<<name<<".txt";
    detector.saveDatabase(m_dbfile, key_name.str(), desc_name.str() );
  }
  
  cout<<"Press esc to exit.."<<endl;
  while((cv::waitKey() & 0xEFFFFF) != 27);
  ofstream out;
  stringstream n;
  n<<"precision_recall_"<<name<<".txt";
  out.open(n.str().c_str(),std::ios::app);
  float pr3,rec3,pr4,rec4,pr02,rec02,pr1,rec1,pr15,rec15;
  evaluate.setThreshold(0.2);
  pr02 = evaluate.getPrecision();
  rec02 = evaluate.getRecall();
  evaluate.setThreshold(1);
  pr1 = evaluate.getPrecision();
  rec1 = evaluate.getRecall();
  evaluate.setThreshold(1.5);
  pr15 = evaluate.getPrecision();
  rec15 = evaluate.getRecall();
  
  evaluate.setThreshold(3);
  pr3 = evaluate.getPrecision();
  rec3 = evaluate.getRecall();
  evaluate.setThreshold(4);
  pr4 = evaluate.getPrecision();
  rec4 = evaluate.getRecall();
  evaluate.setThreshold(5);
  out<<"a="<<alpha<<","<<"k="<<k<<","<<pr3<<","<<rec3<<","<<pr4<<","<<rec4<<","
    <<evaluate.getPrecision()<<","<<evaluate.getRecall()<<","<<evaluate.getMeanError()
    <<","<<pr02<<","<<rec02<<","<<pr1<<","<<rec1<<","<<pr15<<","<<rec15<<endl;
  out.close();
}

// ---------------------------------------------------------------------------

template<class TVocabulary, class TDetector, class TDescriptor>
void LoopDetector<TVocabulary, TDetector, TDescriptor>::readPoseFile
  (const char *filename, std::vector<double> &xs, std::vector<double> &ys,
  std::vector<string> &tss)
  const
{
  xs.clear();
  ys.clear();
  tss.clear();

  fstream f(filename, ios::in);
  string s;

  while(!f.eof())
  {
    getline(f, s);
    if(!f.eof() && !s.empty())
    {
      std::istringstream ss(s);
      std::string token;
      vector<string> values;
      // read from pose file: timestamp, id, x, y, yaw
      while(std::getline(ss, token, ','))
        values.push_back(token);

      xs.push_back(atof(values[2].c_str()));
      ys.push_back(atof(values[3].c_str()));
      std::istringstream sl(values[0]);
      // get only integer part
      std::getline(sl, token, '.');
      tss.push_back(token);
    }
  }
  f.close();
}

// ---------------------------------------------------------------------------

#endif
