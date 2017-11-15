#**Traffic Sign Recognition** 

##Writeup Template

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)


* Explore, summarize and visualize the data set


* Design, train and test a model architecture


* Use the model to make predictions on new images


* Analyze the softmax probabilities of the new images


* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./reprot_pics/visual.png "Visualization"
[image2]: ./reprot_pics/grayscale.png "Grayscaling"
[image3]: ./reprot_pics/augment_pic.png "augment_pic"
[image4]: ./download_images/20kmph.jpg "Traffic Sign 1"
[image5]: ./download_images/no_entry.jpg "Traffic Sign 2"
[image6]: ./download_images/no_truck_passing.jpg "Traffic Sign 3"
[image7]: ./download_images/right_turn.jpg "Traffic Sign 4"
[image8]: ./download_images/stop.jpg "Traffic Sign 5"



###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The line chart shows the percentage of each class in train and validation sets. The bar chart indicates the frequency of percentage distribution in train and validation sets.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)``

As a first step, I decided to convert the images to grayscale because there is too much noise in colored images.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a second step, I confined the pixel to [0,1], because the CNN algorithm requires data that on a 0-1 scale.

As a last step, I subtracted the mean of pixels to so that they are centered around 0.

I decided to generate additional data because I found the the LeNet had overfitted--the validation accuracy was much lower than the training accuracy. So I thought I may need more data.

To add more data to the the data set, I randomly used the following techniques, like gaussian blury, rotation, shift, rotation then shift on the training set, to increase the diversity of data.

Here is an example of an original image and an augmented image:

![alt text][image3]




####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x30 	|
| RELU					|												|
| Max pooling	 2x2     	| 2x2 stride,  valid padding, outputs 14x14x30 				|
| Convolution 5x5	    |  1x1 stride, valid padding, outputs 10x10x50       									|
| RELU					|												|
| Max pooling	 2x2     	| 2x2 stride,  valid padding, outputs 4x4x50 				|
| Convolution 3x3	    |  1x1 stride, valid padding, outputs 2x2x80       									|	
| RELU					|	
| Fully connected		| input 320, output 120 ,dropout keep_prob=0.5      									|
| RELU					|	
| Fully connected		| input 120, output 43 logits      									|
| Softmax				| input logits, output probabilities        									|
|			cross-entropy			|	input	probalilities, output one-hot labels										|
												
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.


I used the liner model to turn the input into logits,then used the softmax to turn the logits into propobilities, then used one-hot coding and cross entry to convert the biggest propobilities into certain labels. 
Besides, I optimized the cross-entropy loss. To balance the training time and performance, I set batch size as 128, number of epochs as 20. I also employed learning rate dacay, so the learning ranged from 0.001 to 0.0005. 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* training set accuracy of 0.962
* validation set accuracy of 0.960 
* test set accuracy of 0.942

	I chose the LeNet-5 solution from the lecture as the first architecture, because the traffic sign image has the same size with the mnist image. So I thought the LeNet-5 might be a good prototype.

	The final accuracy on the training set was almost 0.07 higher than that on the validation set, which indicated the net was overfitted. In addition, the final train accuracy was about 0.90, not good enough. 

	To overcome the overfiffing problem, I augmented the data set by adopting gaussian blury, rotation, shift. As a result, the accuracy gap between the training and validition sets shrinked to around 0.004. However, the accuracy was below 0.92.
	

	I increased the filter numbers, added another convolution layer. Because I thought the traffic sign image is more complex than the mnist image. Finally, it worked!
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because there are few examples of this class, obviously by the analysis of the train/ validation distribution plot.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

|			  Prediction      |     Image	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection      		| Speed limit (20km/h)  									| 
| No entry   			| No entry 										|
| No passing for vehicles over 3.5 metric tons					| No passing for vehicles over 3.5 metric tons											|
| Turn right ahead	      		| Turn right ahead					 				|
| Stop		| Stop      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. It is lower than test set accuracy of 0.942

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the <font color=red>12th cell</font> of the Ipython notebook.

For the first image, the model is not sure that this is a 'Speed limit (20km/h)' sign (probability of 0.17), and the image does contain a 'Speed limit (20km/h)' sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .83         			| Speed limit (30km/h)   									| 
| .17     				| Speed limit (20km/h) 										|
| .0003					| Priority road											|
| .00015	      			| Speed limit (80km/h)					 				|
| .000013				    | Roundabout mandatory     							|

[1  0 12  5 40]

For the second image, the model is pretty sure that this is a 'No entry' sign (probability of 1), and the image does contain a 'No entry' sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| No entry   									| 
| 6.16e-10    				| No passing for vehicles over 3.5 metric tons 										|
| 6.82e-12					| No passing											|
| 2.54e-12	      			| Roundabout mandatory					 				|
| 4.80e-13				    | Go straight or left      							|
[17 10  9 40 37]


For the third image, the model is relatively sure that this is a 'No passing for vehicles over 3.5 metric tons' sign (probability of 0.98), and the image does contain a 'No passing for vehicles over 3.5 metric tons' sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.98        			| No passing for vehicles over 3.5 metric tons   									| 
| 0.0236    				| End of no passing by vehicles over 3.5 metric tons 										|
| 5.09e-09					| Speed limit (80km/h)											|
| 2.95e-09	      			| Speed limit (100km/h)				 				|
| 9.42e-14				    | No passing     							|
 [10 42  5  7  9]
 
For the forth image, the model is pretty sure that this is a 'Turn right ahead' sign (probability of 0.998), and the image does contain a 'Turn right ahead' sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.998         			| Turn right ahead   									| 
| 1.89e-04    				| No passing for vehicles over 3.5 metric tons										|
| 4.16e-06					| Right-of-way at the next intersection											|
| 2.54e-06	      			| End of no passing by vehicles over 3.5 metric tons				 				|
| 2.32e-06				    | Yield   							|

[33 10 11 42 13]

For the fifth image, the model is pretty sure that this is a 'Stop' sign (probability of 0.998), and the image does contain a 'Stop' sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99998         			| Stop sign   									| 
| 2.12e-05    				| Yield 										|
| 1.98e-07					| Turn right ahead											|
| 8.91e-08	      			| Ahead only					 				|
| 2.22e-08				    | No entry     							|

[14 13 33 35 17]




