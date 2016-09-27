# Place_Recognition
Appearance-Based Place Recognition

In this project two already implemented libraries called DBoW and DLoopDetector are used. 
The former library contains a bag of visual words model implementation while the latter uses loop detection methods
for identifying loop closures.

The task of this system is to identify locations utilizing visual information. In other words, based on it’s
current knowledge obtained from a previous experience (e.g a traversed trajectory in a park or any other indoor
or outdoor environment), the system should be able to identify if a location has been previously visited or not. 
To represent and compare locations, captured frames are transformed into a bag of words representation with the 
use of DBoW library.

After obtaining the most similar frames to a given frame, the system performs temporal and 
geometrical checks implemented in DLoopDetector library to identify if a loop closure exists (a location that has 
been visited before). If the system identifies a loop closure, a match representing the same location from a 
previous visit should be returned. Once a match is detected, the system localizes against a map and identifies 
it’s position relative to that map.
