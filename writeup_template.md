# **Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Dataset Summary

I used the pandas librar=y to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The size of the validation set is 4410
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Classes/Labels Distribution

In the bar charts bellow the X-axis show the code for the class (the meaning of the codes can be found here) and the Y-axis is the amount of images for that class. It is paramount for a good training/testing/validation that the distribution of the image classes follows the same pattern. 

![alt text][image1]

### Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because first it decreases the amount of data being managed by a a factor of 2 (the images goes from 32x32x3 to 32x32x1). Second, I found out that using gray images leads to a better model accuracy, what was also seem in this [article]() by Pierre Sermanet and Yann LeCun. I suppose it is somewhat intuitive that colors are not important for this aplication, by eye we can identify the signs with color as well as in grayscale.  

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

Normalizing the data is a good practice because it keeps [numerical stability](http://mathworld.wolfram.com/NumericalStability.html) of the algorithm and having features with a 0 mean and equal variance gives a **Well Conditioned Problem** for the optimizer. The normalization formula is diplayed bellow.

```math
a + b = c
```


By generating adittional data I was able to boost the quality of the model. Adittional 5 images were generated for each image in the training set by aplying a random perturbation to it.

Here is an example of an original image and an augmented image:

![alt text][image3]
![alt text][image3]


##### The difference between the original data set and the augmented data set

* Images were translated by a random amount of pixels vertically and horizontaly in the interval [-2, 2] 
* Images were tilted by a random degree amount in the interval [-15, 15]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model (LeNet) consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Fully connected		| input 400, output = 120						|
| RELU					|												|
| Dropout				| keep probability = 0.5						|
| Fully connected		| input 120, output = 84						|
| RELU					|												|
| Dropout				| keep probability = 0.5						|
| Fully connected		| input 84, output = 43 						|
| RELU					|												|
| Softmax				| 	        									|
 

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the [Adam Optimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer) and used its default learning rate (0.001). Using a NVIDIA Titan X to train the model allowed me to use a big batch size, however in my Jupyter notebook the batch is set for 128 (someone might try on an older GPU). No increase in the performance were observed above 40 **epochs**.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* test set accuracy of 95.3%
* validation set accuracy of 96.7%
 

### Testing random images found on Google


#### Speed limit (70km/h)

![alt text][image8]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0 					| Speed limit (70km/h)  						| 
| 5.1180215e-35			| Speed limit (30km/h) 							|
| Almost 0.0			| Speed limit (20km/h)							|
| Almost 0.0			| Speed limit (50km/h)			 				|
| Almost 0.0	   		| Speed limit (60km/h)     						|


#### Children crossing

![alt text][image8]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.73999822			| Children crossing  							| 
| 0.25391123			| Bicycles crossing 							|
| 0.0060873665			| Road work										|
| 2.5763277e-06			| Beware of ice/snow			 				|
| 5.7231995e-07		    | Slippery road      							|


#### Road work

![alt text][image8]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0 					| Road work  									| 
| Almost 0.0			| Speed limit (20km/h) 							|
| Almost 0.0			| Speed limit (30km/h)							|
| Almost 0.0			| Speed limit (50km/h)			 				|
| Almost 0.0	   		| Speed limit (60km/h)     						|


#### Stop

![alt text][image8]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0 					| Stop 											| 
| 2.1838753e-08			| No entry 										|
| 6.8094623e-09			| Keep right									|
| 1.3789767e-09			| Turn right ahead				 				|
| 1.8331527e-11	   		| Yield    										|


#### Double curve

![alt text][image8]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9999603 			| Double curve			 						| 
| 3.9675648e-05			| Right-of-way at the next intersection 		|
| 3.0263076e-09			| Wild animals crossing							|
| 2.3719895e-09			| Road work			 							|
| 2.2993707e-09	   		| Slippery road    								|


#### Go straight or right

![alt text][image8]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99594325			| Go straight or right  						| 
| 0.0034704579			| Keep right 									|
| 0.00051441899			| Turn left ahead								|
| 6.8129419e-05			| Ahead only			 						|
| 2.9697715e-06	   		| Yield    										|


#### Roundabout mandatory

![alt text][image8]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99163145 			| Roundabout mandatory 							| 
| 0.0081623141			| Speed limit (100km/h) 						|
| 0.00019325051			| Priority road									|
| 9.4156412e-06			| No entry			 							|
| 2.9617011e-06	   		| Keep left     								|


The model seems to work very well for the images selected other than the "Children Crossing" image, which accordingly to the prediction has a 25% chance of being "Bicycles crossing". The could be a raction of many factors as the lack of smaller amount of "Children Crossing" provided for training the model when comparing to the other classes as well to the fact that these two classes seems to have a lot of pixels in common.

