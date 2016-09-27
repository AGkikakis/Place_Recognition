/**
 * File Evaluation.h
 * Date: July 2016
 * Author: Antony Gkikakis
 * Description: Evaluates the perfomance of the DLoopDetector algorithm
 *
 * This file is usedd for evaluating the performance of the DLoopDetector
 * algorithm.
 * It stores the coordinates of the location of each frame as also the location 
 * of the matched frame and determines based on the euclidian distance if the 
 * match is valid.
 * Generates the precision and the recall of the algorithm.
 */
#ifndef __EVALUATION__
#define __EVALUATION__

#include <vector>
#include <stdlib.h>
#include <math.h>

using namespace std;

class Evaluation
{

public:
  struct point
  {
    float x;
    float y;
  };
private:
  // thershold denoting the radius of the circle for the point to be accepted
  // as correct match
  float threshold;
  // number of frames of the experiement
  int n_frames_to_match;
  // contains the x,y coordinates of the matched frame
  vector<point> matched;
  // contains the x,y coordinates of the given frame
  vector<point> current;
  // number of flase positives
  int false_negatives;
public:

  /**
   * Constructor
   */
  Evaluation(const float &threshold = 1.5f, const int &n_frames_to_match = 100 );

  /**
   * Adds the points to the two corresponding vectors
   * @point a: the coordinates where the algorithm detected a loop
   * @point b: the coordinates where the algorithm matched
   */
  void addPoints(const point &a, const point &b);

  /**
   * Returns the precision of the algorithm
   * @return : #(true positives) / (#true positives + #false positives)
   */
  float getPrecision();

  /**
   * Returns the recall of the algorithm
   * @return : #(matched frames) / #(frames)
   */
  float getRecall();

  /**
   * Returns the average of the position error
   * @return : (sum of error) / #(matched frames)
   */
  float getMeanError();

  /**
   * Sets the overal number of frames of the experiement
   */
  void setNumberOfFrames(int n_frames_to_match);
  
  /**
   * @return the number of true positives
   */
  int getTruePositives();

  /**
   * Prints to the screen the performance of the algorithm
   */
  void toString();

  /**
   * Function used to calculate the distance between two points
   * @return float representing euclidian distance between points
   */
  inline float getDistance(const point &a, const point &b)
  {
    return sqrt( pow(( a.x - b.x ),2) + pow(( a.y - b.y ),2) );
  }
  
  /**
   * @return the number of frames the algorithm recalled
   */
  inline int getRecalledFrames()
  {
    return matched.size();
  }
  
  /**
   * Function that decides if the matched pair is valid
   * @return true if the distance between the point and the matched point
   * is less than the threshold
   */
  bool isValidMatch(const point &a, const point &b);
  
  /**
   * @return a number denoting the percent of loop detections the algorithm made
   */
  float getUndetectedPercentage();
  /**
   * Sets the threshold that is used to decide if two points are close to 
   * each other
   */
  inline void setThreshold(const float &threshold)
  {
    this->threshold = threshold;
  }

  /**
   * @return the threshold value
   */
  inline float getThreshold()
  {
    return threshold;
  }
  
  /**
   * increase the number of false negatives
   */
  inline void foundFalseNegative()
  {
    false_negatives++;
  }
  
  /**
   * @return the number of false negatives
   */
  inline int getFalseNegatives()
  {
    return false_negatives;
  }
};// End of header

// ---------------------------------------------------------------------------

Evaluation::Evaluation(const float &threshold, const int &n_frames_to_match)
{
  this->threshold = threshold;
  ( n_frames_to_match!=0 ) ? (this->n_frames_to_match = n_frames_to_match) : (this->n_frames_to_match = 100);
  current.reserve(n_frames_to_match);
  matched.reserve(n_frames_to_match);
  false_negatives = 0;
}

void Evaluation::addPoints(const Evaluation::point &a, const Evaluation::point &b)
{
  current.push_back(a);
  matched.push_back(b);
}

float Evaluation::getPrecision()
{
  return ( (matched.size() != 0) ? (float)getTruePositives()/(float)matched.size() : 0 );
}

float Evaluation::getRecall()
{
  unsigned int t_p = getTruePositives();
  return (float)t_p/ ( (float)(t_p) + (float)(getFalseNegatives()) );
}

float Evaluation::getUndetectedPercentage()
{
  return (float)matched.size()/(float)n_frames_to_match;
}

float Evaluation::getMeanError()
{
  float mean = 0.0;
  for (unsigned int i = 0; i < matched.size(); ++i)
    mean = mean + getDistance(current[i], matched[i]);
  return mean/(float)(matched.size());
}

void Evaluation::setNumberOfFrames(int n_frames_to_match)
{
  this->n_frames_to_match = n_frames_to_match;
  current.reserve(n_frames_to_match);
  matched.reserve(n_frames_to_match);
}

int Evaluation::getTruePositives()
{
  unsigned int true_positives = 0;
  for (unsigned int i = 0; i < matched.size(); ++i)
  {
    if ( isValidMatch(current[i], matched[i]) )
      true_positives++;
  }
  return true_positives;
}

bool Evaluation::isValidMatch(const point &a, const point &b)
{
  return ( getDistance(a,b) < threshold );
}

void Evaluation::toString()
{
  cout<<"Precision of the algorithm is : "<<getPrecision()<<endl;
  cout<<"Threshold for ground truth is : "<<getThreshold()<<endl;
  cout<<"Recall of the algorithm is : "<<getRecall()<<endl;
  cout<<"Frames recalled = "<<getRecalledFrames()<<endl;
  cout<<"Mean error (in meters) of matched frames is : "<<getMeanError()<<endl;
}

#endif
