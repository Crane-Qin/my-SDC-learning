## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---
	
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.jpg
[image2]: ./output_images/HOG_example.jpg
[image3]: ./output_images/sliding_window_boxes.jpg
[image4]: ./output_images/sliding_window_result.jpg
[image5]: ./output_images/bboxes_and_heat.jpg
[video1]: ./vehicle-detected-result.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.


I defined a function `get_hog_features`, applying `skimage.hog()`. Parameters like `orientations`, `pixels_per_cell`, and `cells_per_block`, could be tuned. I also defined a function `extract_features`, where I could also convert color spaces.

I then load all the `vehicle` and `non-vehicle` images and applied `extract_features` on instances of these two classes. 

![alt text][image1]

Here is an example using HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters as below:

| Flip     | Colorspace | Orientations | Pixels Per Cell | Cells Per Block | HOG Channel | Spatial Size |Hist Bin|Accuracy|
| :-----:  | :--------: | :----------: | :-------------: | :-------------: | :---------: | ------------:|----:   |----:   |
| NO       | LUV        | 9            | 8               | 2               | 0           | False        |False   |0.9516  |
| NO       | LUV        | 9            | 8               | 2               | 0           | True,(16, 16)         |False   |0.9744  |
| NO       |   LUV      | 14           | 16              | 2               | 0           | True,(16, 16)       |True,32  |0.9811  |
| NO       | YUV       | 9            | 8               | 2               | ALL          | True,(32, 32)       |True,16 |0.9947  |
| Yes      |  YUV        | 9            | 8               | 2               | ALL          | True,(32, 32)       |True,16 |0.9941  |
| Yes      |  YUV        | 9            | 8               | 2               | ALL          | True,(32, 32)       |False |0.9935  |
| NO       |   LUV      | 14           | 16              | 2               | ALL           | True,(16, 16)       |True,32  | 0.9952 |
| NO       |   LUV      | 10           | 16              | 2               | ALL           | True,(16, 16)       |True,32  | 0.9935 |
| NO       |   YUV      | 11           | 16              | 2               | ALL           | False       | False | 0.9873 |
| NO       |   YCrCb      | 11           | 8              | 2               | ALL           | False       | False | 0.9876 |


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I extracted the cars and non-cars features via the function `extract_features` with the parameter in the chart. Normalization was a key step so that a certain subset of the features wouldn't dominate. I used `sklearn.StandardScaler()` method for this. Afterwards I shuffled and splited them into training and test sets, fed the former to the `sklearn.svm.LinearSVC()` and fed the latter to the trained model to get the accuracy.

Finally, I chose parameters in the last row, whose accuracy was 0.9876.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
Considering the relation between the car image size and its distance, I set the scales on different image height.
I drew windows on the test pictures as follows, to see if the windows could cover the cars.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 4 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections across the last 10 frames and  created a heatmap. I also used the threshold on each `find_cars` step to reject false positives. 

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the frames of video:

![alt text][image5]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I set the scaling ratio manually, if the car are very large or very small, the pipeline would fail to detect it. So it is necessary to test on various videos to get the best parameters. 

The accuracy on detecting cars is not high enough. If the number of sliding windows increase largely, the chance of false positives would be enhanced. One possible way to modify the classifier is data augmentation.

What's more, there are plenty of car features other than colors, such as speed, location, and route. If given enough time, I would combine these together to make a robust classifier. 