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

#### 2. Feature extraction from training images

In the next step, feature extraction was applied on some test data. 

The code for feature extraction can be found in **calculate_features.py**. Most of the code is heavily inspired by the Udacity lab code and existing OpenCV algorithms.

I decided to use HOG features, spatial and colour information, since classification using these three kinds of features performed a bit better (see later sections). 

Here is an example using the HSV colour space and HOG parameters of 18 orientation bins, cell size of 8 x 8 pixels and a block size of 2 x 2 cells:

![HOG](/writeup_images/HOG.JPG)

#### 3. Final choice of feature extraction parameters

##### 3.1 HOG parameters

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

The only remaining factor is the colour space. I visualized the HOG features of a random car in HSV, HLS and LAB colour spaces. The three channels of HSV seem to be capturing different features in each channel, so I opted for this colour space. HSV, HLS and LAB seem to perform quite well with a linear SVM classifier, so I although I didn't investigate this thoroughly, I presume that results will be more or less similar with the other aforementioned colour spaces.

![HogColourSpaces](/writeup_images/HogColourSpaces.JPG)

This results in a feature vector of size 3528 (7 x 7 x 2 x 2 x 18).

##### 3.2 Colour histogram

The colour histogram per 64x64 frame has been parameterized such that it has 32 bins for each channel (so 96 features in total)

##### 3.3 Spatial binning of colour

Each 64x64 frame is rescaled to 16x16, so in total 16x16x3=768 features are generated.

#### 4. Training of the classifier using the selected features.

##### 4.1 Hyper-parameter search

First of all, all the car and non-car data is randomly shuffled and for each image the features are extracted using the aforementioned parameters. Then, all data is normalized with a *StandardScaler* from the **sklearn.preprocessing** library. Finally, the data is split into a training set (80%) and a test set (20%) for cross-validation. The car images are labelled with '1' and the non-car images are labelled with '0'. The code can be found in *prepare_hog_data()* in **svm_classify.py**

I did a grid search (*GridSearchCV()*) to find an optimal SVM. Following hyper-parameters were investigated:

* RBF kernel:
  * Gamma:	0.001 and 0.0001
  * C: 1, 10, 100, 1000
* Linear kernel:
  * C: 1, 10, 100, 1000

The results for the hyper-parameter set when optimizing for precision were:

| Kernel type | C      | Gamma      | Grid score |
| ----------- | ------ | ---------- | ---------- |
| RBF         | 1      | 0.001      | 0.972      |
| RBF         | 1      | 0.0001     | 0.974      |
| RBF         | 10     | 0.001      | 0.974      |
| **RBF**     | **10** | **0.0001** | **0.982**  |
| RBF         | 100    | 0.001      | 0.974      |
| RBF         | 100    | 0.0001     | 0.982      |
| RBF         | 1000   | 0.001      | 0.974      |
| RBF         | 1000   | 0.0001     | 0.982      |
| Linear      | 1      | N/A        | 0.972      |
| Linear      | 10     | N/A        | 0.972      |
| Linear      | 100    | N/A        | 0.972      |
| Linear      | 1000   | N/A        | 0.972      |

The results for the hyper-parameter set when optimizing for accuracy were:

| Kernel type | C      | Gamma      | Grid score |
| ----------- | ------ | ---------- | ---------- |
| RBF         | 1      | 0.001      | 0.966      |
| RBF         | 1      | 0.0001     | 0.975      |
| RBF         | 10     | 0.001      | 0.967      |
| **RBF**     | **10** | **0.0001** | **0.982**  |
| RBF         | 100    | 0.001      | 0.967      |
| RBF         | 100    | 0.0001     | 0.981      |
| RBF         | 1000   | 0.001      | 0.967      |
| RBF         | 1000   | 0.0001     | 0.981      |
| Linear      | 1      | N/A        | 0.973      |
| Linear      | 10     | N/A        | 0.973      |
| Linear      | 100    | N/A        | 0.973      |
| Linear      | 1000   | N/A        | 0.973      |

The SVM classifier hyper-parameter search was performed on the HOG features only. 5-fold cross validation was used and the search was spread out over 4 jobs (4 CPU's).

##### 4.2 SVC training

We trained a SVM classifier with RBF kernel with gamma= 0.0001 and C=10. This resulted in an accuracy on the test data of 99.37%. The colour and spatial data gave more than 1% extra accuracy. Also, the *probability* parameter has been set to *True*, so that we can do predictions with probabilities during the sliding window search.

Some tests were run with a SVC with only HOG features (so no colour or spatial data was used). Some lane lines and guardrails get high probabilities, due to the fact that they sometimes have the same (strong) gradients and thus colour and spatial data was necessary to have a better classification.

The trainer classifier is stored in the pickle file **"all-features-rbf-svm.p"**

The scaler used to normalize all the features is stored in the pickle file **"all-features-scaler.p"**

The code can be found in *prepare_training_data()* and *train_svm()* in **svm_classify.py**

### Sliding Window Search

#### 1. Scale, region of interest and window overlap parameters

I defined the following parameters for the sliding window search:

- Cars further away (small projections): 
  ROI: (550,370)  - (1024,498), window size: 64x64 (scale 1), overlap: 50%
  ![Window64x64](/writeup_images/Window64x64.JPG)
- Cars in middle range distance:
  ROI: (530,390)  - (1144,534), window size: 96x96 (scale 1.5), overlap: 50%
  ![Window96x96](/writeup_images/Window96x96.JPG)
- Cars close by: 
  ROI: (480,400)  - (1280,592), window size: 128x128 (scale 2), overlap: 50%
  ![Window128x128](/writeup_images/Window128x128.JPG)
- Cars passing by on the right side:
  ROI: (944,380)  - (1280,620), window size: 192x160 (scale (3,2.5) ), overlap: 75%
  ![Window192x160](/writeup_images/Window192x160.JPG)
- Cars passing by on the right side (2):
  ROI: (896,396)  - (1280,636), window size: 256x192 (scale (4,3) ), overlap: 75%
  ![Window256x192](/writeup_images/Window256x192.JPG)

Note that the sliding window area is tuned optimally for this video and set manually. This was tuned this way for optimal detection and reduce false positives (e.g. guardrails, oncoming cars)

Also note that the sliding window algorithm has been adapted so **that a different scale can be applied for the X and Y directions**. This is useful for situations where the car is visible from one of its sides as well (see the black car in the picture above). The training data contains also test images where cars have been rescaled so that the aspect ratio is not respected anymore.

#### 2. Pipeline

The pipeline can be found in **process_frame.py**. The same pipeline is executed for each image/frame.

- Convert from RGB to HSV
- For each scale in the sliding window set:
  - Rescale the frame to 64 x 64
  - Calculate the HOG features
  - For each sliding window with the same scale:
    - Calculate the colour histogram and the spatial colour bins
    - Make a feature vector containing the colour histogram, the spatial colour data and the subset of the HOG features that apply to the sliding window
  - Predict with the SVM classifier if the feature vector corresponds with a car or not (with *SVC.predict_proba()*. **Only when the probability is higher than 0.7**, then the sliding window is accepted as a candidate car (the bounding box and the probability are stored as *BoundingBox* instances). A few examples from the resulting pipeline so far, are given here:
    ![BoundingBox1](/writeup_images/BoundingBox1.JPG)
    ![BoundingBox2](/writeup_images/BoundingBox2.JPG)
- Create a heat map that will be constructed using the candidate windows from the current frame and the heat maps from the past 6 frames (see **heatmap.py**)
  - Initialize the current heat map (same size as the frame, with a single "heat" channel) with zeroes
  - For each candidate window (*BoundingBox* instance) from the current frame:
    - For all positions within the candidate window in the heat map:
      - Set the current "heat" pixel value to the **maximum of its current value and the probability predicted by the SVM classifier**
  - Add the heatmaps from the past 6 frames to the heatmap from the current frame to create a cumulative heatmap
- Apply a threshold on the cumulative heatmap: all "heat" pixels with pixel value > 5.5 are accepted as pixels belonging to a car
  ![Heatmap1](D:\Projects\SDC\CarND-Vehicle-Detection\writeup_images\Heatmap1.JPG)

  ![Heatmap2](D:\Projects\SDC\CarND-Vehicle-Detection\writeup_images\Heatmap2.JPG)

  ![Heatmap3](D:\Projects\SDC\CarND-Vehicle-Detection\writeup_images\Heatmap3.JPG)

  ![Heatmap4](D:\Projects\SDC\CarND-Vehicle-Detection\writeup_images\Heatmap4.JPG)

  ![Heatmap5](D:\Projects\SDC\CarND-Vehicle-Detection\writeup_images\Heatmap5.JPG)

  ![Heatmap6](D:\Projects\SDC\CarND-Vehicle-Detection\writeup_images\Heatmap6.JPG)

  ![Heatmap7](D:\Projects\SDC\CarND-Vehicle-Detection\writeup_images\Heatmap7.JPG)
- Construct a bounding box around all neighbouring pixels with non-zero values in the thresholded cumulative heatmap and label them separately.


---

### Video Pipeline

The video pipeline can be found in **process_video.py**

The output of the video pipeline on the project video can be [found here](./project_output.mp4)

Elimination of false positives has been discussed in the frame pipeline.

---

### Shortcomings and possible improvements

- False positives could be further eliminated hard negative mining. In that case, we could take all sliding windows with high probabilities (> 0.5 or >0.7) and add them to the (non vehicle) training data to retrain the SVM classifier. 
- The video pipeline is much too slow. It could be optimized by having GPU implementations of feature extraction algorithms. Also, the number of HOG features could be reduced (e.g. 9 orientations bins instead of 18 bins). A linear SVM classifier could be used as well, as the test accuracy statistics were also quite good (in the hyper-parameter grid search). The linear SVM classifier trains faster and also predicts faster, but a RBF kernel has been used in this project to dig a bit deeper into the course material. 
- The ROI's of sliding window search are manually optimized for the project video and can not be generalized to other videos. The ROI's should actually be set up automatically, based on other data (for example the lane detection, but also other computer vision techniques)
- Finally, sensor fusion could also aid in the detection of cars. Lidars, radars, stereo vision etc. could give more information about the actual scene. 



