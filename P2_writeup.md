## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

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

[image1]: ./output_images/undistorded_chessboard.jpg "Undistorted"
[image2]: ./output_images/undistorded_example.jpg "Road Transformed"
[image3]: ./output_images/contribs.jpg "Binary Example"
[image4]: ./output_images/warping.jpg "Warp Example"
[image5]: ./output_images/fitted.jpg "Fit Visual"
[image6]: ./output_images/example_output.jpg "Output"
[video1]: ./output_videos/output_project.mp4 "Video"
[image7]: ./output_images/failed.jpg "Failed lane detection"


## Rubric Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.
- Camera Calibration
- Image Pipeline
  - Distorsion correction
  - Thresholded binary image
  - Perspective transform
  - Lane pixels identification
  - Radius of curvature calculation
  - Output result
- Video Pipeline
- Discussion

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./P2.ipynb".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to an image used for the calibration with the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The code to demonstrate this step in included in the cell 2 and 3 of the notebook. I simply use the calibration matrix obtained through the `cv2.calibrateCamera()` and apply it to a test image. The undistorded output looks like this one:

![alt text][image2]

If we zoom in a bit, we can see that the traffic sign on the left looks a bit flattened.


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

In order to generate a binarized image that will help us detect lanes robustly, I implemented a combination of color and gradient thresholds. The code is detailed in the `image_utils.py` file (**from line 86 to 136**). I implemented the following pipeline:
- Apply a hard thresholding on the **Sobel operator**
- Apply a gradient magnitude thresholding
  - Both previous thresholding are enough to get a robust identification of image areas with a strong change in color. Note that we are looking for light colored (most of the time white or yellow) on a dark background : asphalt, concrete... Hence, the gradient around the lines is supposed to be strong.
- Apply gradient direction thresholding, although I believe this step is not as insightful than the previous two. We can set it to `False` in my function.
- Transform image in the HLS color space and apply S channel thresholding

Below is an example of how my gradient and color thresholding contributes to my output, together with the final binarized image.

![alt text][image3]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in **lines 139 through 148** in the file `image_utils.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[0.162*image.shape[1], image.shape[0]],
                  [0.86*image.shape[1], image.shape[0]],
                  [0.558*image.shape[1],0.65*image.shape[0]],
                  [0.444*image.shape[1], 0.65*image.shape[0]]])

dst = np.float32([[image.shape[0]*0.4, image.shape[0]],
                  [image.shape[1]*0.8, image.shape[0]],
                  [image.shape[1]*0.8, 0],
                  [image.shape[0]*0.4, 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 207.36, 720      | 288, 720        |
| 568.32, 468.0      | 1024, 720      |
| 1100.8, 720     | 1024, 0      |
| 714.24, 468.0      | 288, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image and here is the result:

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I implemented a method `find_lane_pixels()` ** (line 151 to 230 from `image_utils.py`)**. This method searches for two peaks in histogram, one on the left part of the picture and the other one on the right part. This will, in most cases, highlight lanes marking. Then, we can fit a polynomial to these detected lanes. I used two methods:
- The first one `fit_polynomial()` ** (line 233 to 261 from `image_utils.py`)** will simply fit a polynomial given `x` and `y` coordinates.
- The second one `search_around_poly()` ** (line 264 to 317 from `image_utils.py`)** will use a previously fitted polynomial to update the polynomial equation of the lane. Below is an example of fitted line.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I implemented the `measure_curvature()` method in **line 320 to 344 of `image_utils.py`**. This method uses an approximation of the distance in meters per pixel and the radius formula to estimate the radius of curvature.
This method also returns the approximated distance from the center of the lane.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Finally, I implemented all of the above step in **cell [21] of the p2.ipynb jupyter notebook** in the function `process_image()` after having implemented a few sanity checks together with the `Line()` class. This class enables us to track progress on our lane detections and know when detections fail. Moreover, using recent tracking progress we can discard complicated frames, although the problem remains if we lose detections for a too long period of time.

Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's my [result](./output_videos/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

To be able to detect lanes I implemented gradient and color thresholding methods, followed by warping method to create a "bird-eye" view to be able to estimate the lane curvature.

In my approach, there are several drawbacks that are highlghted by the challenge and harder challenge videos.
Although my current algorithm does not seem to deviate that much in the challenge, I can see it has a lot of trouble detecting lanes when there is noise on the lane marking (for instance the black line on top of the white one on the first frames of the challenge). It also has difficulty differenciating between shadow on the left of the barrier and the lane marking.

To robustify my algorithm to this problem I am thinking of several leads:
- Sanity checks could be more rigid.
- We could also try to play with more thresholds, and try using some more channels. Indeed, I only used the S channel.
- The histogram peaks uses a argmax on the left and right. However, in the case of strong shadow parallel to the yellow line, we can mistake the shadow for the line (this is what happens actually for left lane of the challenge video)
- I did not implemented a region mask and this could remove some noise in the picture.
- All thresholding is fixed at the moment and improving further could induce adaptative thresholding depending on light conditions and some noise removal due to high lighting.

An instance of the last bullet point is provided by the `harder_challenge_video.mp4`.
Below is a frame of the harder challenge where there is a lot of variations in lighting. We can see that the histogram will be completely messed up by the saturation and that the lane finding method based on histogram will fail.

![alt text][image7]

Hence, adding some more channels thresholding, and manipulations together with an improved lane finding method (not the basic histogram) may be necessary.
