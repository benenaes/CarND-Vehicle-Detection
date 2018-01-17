---
typora-root-url: ./
---

## Writeup
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

### Histogram of Oriented Gradients (HOG)

#### 1. Training data

My training and cross-validation data was the set of images in the **vehicle** and **non-vehicle** images provided by Udacity. The code to load all images can be found in **load_samples.py**.

The code for the **Kitti** car images and non-vehicle images is straight forward. 

![training_data](/writeup_images/training_data.JPG)

Since there were a lot of time series based images in the **GTI** car image database, I decided to apply a **template matching** technique in **find_unique_gti.py**. I constructed a list of semi-unique car images by making a patch (middle 50%) of the first image. If subsequent images have a template matching error below a certain threshold, then they are considered as "similar images". If an image has a template matching error above the aforementioned threshold, then the image is added to the list of unique car images and a new patch is made from that image (to be used in the template matching algorithm on the subsequent images). This works reasonably well, since the same car is usually only visible in subsequent images. There are some exceptions where the same car (but a different part) is visible, but not enough to disturb the balance of the vehicle training data.

![TemplateMatching](/writeup_images/TemplateMatching.JPG)

#### 2. HOG feature extraction from training images

In the next step, HOG feature extraction was applied on some test data. 

The code for feature extraction can be found in **calculate_features.py**. Most of the code is heavily inspired by the Udacity lab code and existing OpenCV algorithms.

I decided to restrict the features to HOG features only, since classification on colour and spatial features performed less well (see later sections). Cars can have many different colours, ranging from matte to flashy, so it's normal that classifiers find it hard to discern cars from non-car objects. The case would be different in case of thermal images probably, because local thermal properties might be useful for classification. Also, most literature usually only mentions HOG+SVM. 

Here is an example using the HSV colour space and HOG parameters of 18 orientation bins, cell size of 8 x 8 pixels and a block size of 2 x 2 cells:

![HOG](/writeup_images/HOG.JPG)

#### 3. Final choice of HOG parameters

The major parameters for the HOG feature extraction are:

- Pixels per cell
- Cells per block
- Number of orientation bins
- Colour space
- Block normalization

First of all, I was inspired by the presentation of Navneet Dalal on "Histogram of Oriented Gradients (HOG) for Object Detection" (see https://www.youtube.com/watch?v=7S5qXET179I). From minute 35-45, there is an interesting part about parameter optimization.

The block normalization was advised to be L2-Hysteresis. Also, the number of orientation bins were advised to be 9 (0°-180°) or 18 (0°-360°). 

Optimal cell and block size in the presentation were less clear in the presentation, although the differences in miss rate are not too substantial at least when the block size is not too small and the cell size not too large. I opted for a cell size of 8 x 8 pixels and a block size of 2 x 2 cells.

The HOG parameters are grouped with the *HogParameters* class (see **calculate_features.py**)

The only remaining factor is the colour space. I visualized the HOG features of a random car in HSV, HLS and LAB colour spaces. The three channels of HSV seem to be capturing different features in each channel, so I opted for this colour space. HSV, HLS and LAB seem to perform quite well with a linear SVM classifier, so I presume that results will be more or less similar with the other aforementioned colour spaces.

![HogColourSpaces](/writeup_images/HogColourSpaces.JPG)



#### 4. Training of the classifier using selected HOG features.

First of all, all the car and non-car data is randomly shuffled and for each image the HOG features are extracted using the aforementioned parameters. Then all data is normalized with a *StandardScaler* from the **sklearn.preprocessing** library. Finally, the data is split into a training set (80%) and a test set (20%) for cross-validation. The car images are labeled with '1' and the non-car images are labeled with '0'. The code can be found in *prepare_hog_data()* in **svm_classify.py**

I did a grid search (*GridSearchCV()*) to find an optimal SVM. Following parameters were investigated:

* RBF kernel:
  * Gamma:	0.001 and 0.0001
  * C: 1, 10, 100, 1000
* Linear kernel:
  * C: 1, 10, 100, 1000

### Sliding Window Search

#### 1. Scale, region of interest and window overlap parameters

I defined the following parameters for the sliding window search:

- ROI: (320,370)  - (1024,562), window size: 64x64 (scale 1), overlap: 50%
  ![Window64x64](/writeup_images/Window64x64.JPG)
- ROI: (280,390)  - (1144,582), window size: 96x96 (scale 1.5), overlap: 50%
  ![Window96x96](/writeup_images/Window96x96.JPG)
- ROI: (256,410)  - (1280,602), window size: 128x128 (scale 2), overlap: 50%
  ![Window128x128](/writeup_images/Window128x128.JPG)
- ROI: (944,380)  - (1280,620), window size: 192x160 (scale (3,2.5) ), overlap: 75%
  ![Window192x160](/writeup_images/Window192x160.JPG)
- ROI: (896,396)  - (1280,636), window size: 256x192 (scale (4,3) ), overlap: 75%
  ![Window256x192](/writeup_images/Window256x192.JPG)

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

