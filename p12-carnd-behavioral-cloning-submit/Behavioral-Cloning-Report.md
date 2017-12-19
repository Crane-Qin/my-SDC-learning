#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)


[image4]: ./examples/left_2016_12_01_13_39_02_107.jpg "Left Image"
[image5]: ./examples/right_2017_12_16_13_15_23_568.jpg "Right Image"
[image6]: ./examples/center_2017_12_16_13_19_53_402.jpg "Center Image"
[image7]: ./examples/image_flipped.jpg  "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* Behavioral-cloning-Report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

First, I used a Keras lambda layer (model.py line 93) to normalize the data.

Second, I employed a Cropping layer (model.py line 94) to trim the images, so that there were less irrelevant information,like the sky and the hood of the car.

Third, the model consists of a convolution neural network with 3x3 or 5x5 filter sizes and depths between 24 and 64 (model.py lines 95-99). On each convolution layer, a RELU layer is included to introduce nonlinearity. 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 100). 

I also set the training epochs = 3, to prevent overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 114-118). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 112).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of udacity dataset and my own dataset.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach


My first step was to use LeNet, a familiar model I used. Afterwards, I run the simulator to see how well the car was driving around track one. But it didn't work well. At some point of the track, the car crashed out of the road. 


Then I tried the nVidia Autonomous Car introduced in the course. The model details were discussed previously. In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. When I run the simulator again, it seemed nice, although the car run across the dashed line occasionally. 

To combat the overfitting, I mainly employed a dropout layer and collected more data, especially focusing on those spots.

Finally the vehicle was able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 96-109) was as follows:

| Layer (type)                   |Output Shape        |Connected to     |
|--------------------------------|------------------|-----------------|
|lambda_1 (Lambda)               |(None, 160, 320, 3)  |input   |
|cropping2d_1 (Cropping2D)  |(None, 90, 320, 3)   |lambda_1         |
|convolution2d_1 (Convolution2D) |(None, 43, 158, 24)   |cropping2d_1         |
|convolution2d_2 (Convolution2D) |(None, 20, 77, 36)  |convolution2d_1  |
|convolution2d_3 (Convolution2D) |(None, 8, 37, 48)    |convolution2d_2  |
|convolution2d_4 (Convolution2D) |(None, 6, 35, 64)    |convolution2d_3  |
|convolution2d_5 (Convolution2D) |(None, 4, 33, 64)    |convolution2d_4  |
|dropout_1 (Dropout)             |(None, 4, 33, 64)       |convolution2d_5  |
|flatten_1 (Flatten)             |(None, 8448)             |dropout_1        |
|dense_1 (Dense)                 |(None, 100)         |flatten_1        |
|dense_2 (Dense)                 |(None, 50)           |dense_1          |
|dense_3 (Dense)                 |(None, 10)           |dense_2          |
|dense_4 (Dense)                 |(None, 1)             |dense_3          |



####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first used the data provided by udacity, but it didn't work well.

Thus, I added my own data to the dataset:

**one lap driving forward**
**one lap driving backward**
**one lap focusing on driving smoothly around curves**

To augment the data set, I used images of multiple cameras. 

**For left image, steering angle is added by 0.2**
**For right image, steering angle is subtracted by 0.2**

Left camera image example:	
![alt text][image4]

Left camera image example:	
![alt text][image5]

And I also flipped images and angles:  

Center image:	
![alt text][image6]

Flipped image:		
![alt text][image7]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. At last, my training loss was 0.0033 and validation loss was 0.0147. By employing more ways, the loss gap may be reduced further. But I chose to stop here, for the good prediction result. 
