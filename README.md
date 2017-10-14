# OpenCV_Position
This algorithm calculates the position of a camera from a video feed.
A symbol with known dimensions is extracted from the video and used to create a homography.
The homography is then decomposed into a rotational matrix and translation matrix. 

Algorithm Steps
   1. Get Frame
   2. Gaussian Blur Grey Scale Image
   3. Canny Line Detection
   4. Contour Detection (OpenCV Detect Contour)
   5. Contour Selection 
      1. 12 Sides
      2. Perimeter Threshold 
      3. Internal Angle Threshold 
   6. Homography From Image to Undistorted Symbol 
   7. Remove Camera Matrix
   8. Decompose Homography (Fast Homography Decomposition Zhang)
   
Dependencies
   - Python 2.7
   - Opencv 2
   - Numpy
   - Scipy

Configuration
   - Camera Matrix 
   - Symbol “H”
