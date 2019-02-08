## Advanced Lane Finding
#### By Abeer Ghander

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./camera_cal/calibration1.jpg "calibration image 1"
[image1]: ./output_images/calibration1_undist.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image21]: ./output_images/test1_undist.jpg "Road Transformed Undistorted"
[image3]: ./output_images/test1_binary.jpg "Binary Image"
[image4]: ./output_images/test1_src_roi.jpg "Src Points Image"
[image41]: ./output_images/test1_birdview.jpg "Warp Image"
[image5]: ./output_images/test1_classifiedlanes.jpg "Classified lane lines"
[image6]: ./output_images/test1_final_image.jpg "Output"
[image7]: ./output_images/test5_fromPrior_birdview.jpg "Output"
[image8]: ./output_images/test5_fromPrior_final.jpg "Output"
[image9]: ./output_images/test6_fromPrior_birdview.jpg "Output"
[image10]: ./output_images/test6_fromPrior_final.jpg "Output"
[video11]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in "./advanced_lane_finding.ipynb" in the function `calibrate_camera`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to one of the calibration images using the `cv2.undistort()` function and obtained this result: 

**Original calibration image:**
![alt text][image0]

**Undistorted calibration image:**
![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The code for this step is in the third code cell of the project notebook.

Having the matrix and distortion coefficients of the camera, any input image can be undistorted using the cv2.undistort function. This works on colored or grayscaled images. In our project, the undistortion step is performed on the original colored image.

To demonstrate this step, I will show the result of applying the distortion correction to one of the test images like this one:
**Taking this image as input:**
![alt text][image2]

**After applying the distortion-correction, the output is:**
![alt text][image21]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (Code fourth cell). 

The function `transform_img_to_binary()` transforms the original undistorted image to a binary image using some filters. In the project, I used the color gradients HLS to get filtered output which is not affected by shades or yellow/white lanes. Then, I applied the sobel derivative on the X-axis for the lightness, and filtered out the thresholds. Then, I filtered the saturation color components separately. At the end, the final combined binary image mixes the results ofthe absolute Sobel-X for lightness with the saturation color gradient to form the final binary image which is ready for the next steps.

Here's an example of my output for this step. 

**Binary image for the undistorted image:**
![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is a function called `get_bird_view()`, which appears in the 5th code cell of the IPython notebook.  The `get_bird_view()` function takes as inputs an undistorted binary image (`binary_img`), as well as source (`src`) and destination (`dst`) points. The `cv2.warpPerspective()` is applied on the binary image and the transformation matrix to get the warped image. I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([(582,465),
              (727,465), 
              (280,690), 
              (1100,690)])

offset = 250
dst = np.float32([(offset,0),
              (undist_img.shape[1]-offset,0),
              (offset,undist_img.shape[0]),
              (undist_img.shape[1]-offset,undist_img.shape[0])])
```

This resulted in the following source and destination points:

| Source          | Destination     | 
|:---------------:|:---------------:| 
| (582, 465)      | (250, 0)        | 
| (727, 465)      | (1030, 0)       |
| (280, 690)      | (250, 720)      |
| (1110, 690)     | (1030, 720)     |

Plotting the source points on the undistorted image gives the following:
**Source points on image:**
![alt text][image4]

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

**Birdview image:**
![alt text][image41]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then after obtaining the birdview image, the next step is to identify the pixels belonging to each of the left and the right lines of the lane. This is applied using the sliding window concept for the the birdview image. This is the code in the 6th code cell of the project notebook.

Finding pixels (in the `find_lane_pixels()` function) is done using histogram function for vertical pixels for the lower half of the binary birdview image. In the image, two histogram peaks should be identified. These peaks are mapped to the right and left lane lines. The classification of pixels whether they belong to left or right lane lines, is done by splitting the image to two vertical halves.

Using a sliding window approach, the lane lines are splitted to n number of windows to identify the original region to have meaningful lane markings. After classification of the markings to each side of the lane, and using the sliding windows, a polynomial fitting curve function is performed to identify the line curvature.
![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the function `measure_curvature_real()` which is in the 7th code cell of the project notebook. In this function, given the left fit and right fit for the lane lines, the curvature of the road can be calculated by finding the result of the equation: ((1+(2Ay+B)^2 )^3/2) / |2A| . This equation returns the radius of curvature in pixels. After that, the resulting pixels are converted to meters. 

Then, to identfy the center of the image, assuming the camera to be mounted on the center of the vehicle, the mid point of the lane is identified and the difference from the center of the image is used to find the offset from the middle of the lane.

A helper function `write_data()` is implemented to add the text to images with the curvatures, and offset to the center (to be used in the video processing.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step the 8th code cell of the project notebook in the function `draw_lane()`. It  Warps the image back to the original perspective by calling `warpPerspective` again using inverse matrix `Minv` and draws the lane lines on the original perspective image given the left and right polynomial fits. Here is an example of my result on a test image:

![alt text][image6]

---
#### Calculating lanes given previous polynomial fits for lanes
In case of processing videos, it is expected to have the frame linked to each other. This helps optimizing the search for lane pixels (instead of using the sliding windows approach) and calculation of new polynomial fits for the current frame.

For this purpose, the `search_around_poly()` function is implemented and tested on the image sequence test4, test5 and test6. Where test4 follows the normal pipeline and applies the sliding windows approach, and test5 and test6 are using the `search_around_poly()` function. Here is the sample output for the search from prior function:

**Birdview image for test5.jpg :**
![alt text][image7]

**Final image for test5.jpg :**
![alt text][image8]

**Birdview image for test6.jpg :**
![alt text][image9]

**Final image for test6.jpg :**
![alt text][image10]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

#### Explanation
The video pipelines processes each image of the video by performing the steps: 
1. undistortion
2. binary transformation
3. birdview perspective transformation
4. finding fitted line either using sliding windows or region from previous detection, finding lane width, curvature and offset of ego vehicle to center of the lane, and adding the detected fit to list of fitted lines. 

A class `Line()` is defined to store and processes some common attributes for lane lines such as: `detected`, `best_fit`, `last_fit` and last valid n detections. Class `Line()` has the function `add_fit()` which process the calculated fit in the current frame. It verifies that the calculated lane width is valid, if it was not valid, the last previous fit is taken, and the current detection is invalidated. Also, in case of very sharp curvature for the road lane, the fitted lines are invalidated. 

`best_fit` is the moving average for the last 10 detections for the lines, which ensures minimal jitters between frames.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

**Birdview Image Challenge**
It was challenging to get the birdview image in a good accepted shape. Having it hardcoded values is not efficient. Ideally, the source points should be more dynamic, so that in case of steep curves, the birdview images are not corrupted.

**Saving binary images using openCV**
It was not easy to find out how to save the binary images. In the beginning, it was always a black image, until I used the `normalize()` function with binary images.

**Finding lane lines**
It works well in case of roads with no steep curves, no sun reflections on the camera and no extreme shades on the road. Also, when the variation of colors for the road is not very high. In the challenge_video.mp4, the lane does not have a single color, which confuses the algorithm which lines to select as left or right lane lines. To improve the algorithm, we can limit the window searching region to avoid the middle of the image. Also, it would be good to take the stable line of the lane, and estimate where the other line should be and use this information to better identify the location of the other lane line.

**Processing video, invalidating lines**
It was challenging to decide when to invalidate the lines, and when not. The lane width calculation helped a lot in simplifying the processing to decide whether the polynomial fit is valid or not. Further improvements could be made by verifying the rate of change in the road curvature and the shifting of lane lines.

