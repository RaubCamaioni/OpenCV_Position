# OpenCV_Position
This algorithm calculates the position of a camera from a video feed.
A symbol with known dimentions is extracted from the video and used to create a homography.
The homography is then decomposed into a rotational matrix and translation matrix. 

Algorithm Steps
   1. Get Frame
   2. Gaussian Blur Greysale Image
   3. Canny Line Detection
   4. Countour Detection (OpenCV Dectect Countour)
   5. Countour Selection 
      a. 12 Sides
      b. Perimeter Threshold 
      c. Internal Angle Threshold 
   6. Homography From Image to Undistorted Symbol 
   7. Remove Camera Matrix
   8. Decompose Homography (Fast Homography Decomposition Zhang)
