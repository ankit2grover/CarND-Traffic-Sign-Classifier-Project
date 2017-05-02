#**Traffic Sign Recognition** 

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

You're reading it! and here is a link to my [project code](https://github.com/ankit2grover/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python and pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 4410
* The shape of a traffic sign image is 32,32,3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
It is a bar chart showing how the data is not uniformly distributed across the classes. 
Some classes have  less amount of data as compared to others.

![alt tag] (https://github.com/ankit2grover/CarND-Traffic-Sign-Classifier-Project/blob/master/OutputImages/BarChart.png)

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because grayscale images provided better accuracy than RGB channel images and for detecting signs we need text and shapes data, so it is always better to convert images into grayscale.

Here is an example of a traffic sign image before and after grayscaling.

Original Input image
![alt tag] (https://github.com/ankit2grover/CarND-Traffic-Sign-Classifier-Project/blob/master/OutputImages/OriginalInput.png)

Grayscale image
![alt tag] (https://github.com/ankit2grover/CarND-Traffic-Sign-Classifier-Project/blob/master/OutputImages/GrayscaleInput.png)

As a last step, I normalized the image data in the range of -0.5 to 0.5 because normalization helps in limiting the range of input values and helps in detecting shapes really well.

Normalized Grayscale image
![alt tag] (https://github.com/ankit2grover/CarND-Traffic-Sign-Classifier-Project/blob/master/OutputImages/NormalizedInput.png)



####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 7x7     	| 1x1 stride, Valid padding, outputs 26x26x8 	|
| RELU					| Activation layer for convolution 0			| 
| Convolution 7x7       | 1x1 stride, Valid padding, outputs 20x20x16 	|
| RELU					| Activation layer for convolution 1			| 
| Convolution 7x7       | 1x1 stride, Valid padding, outputs 14x14x32 	|
| RELU					| Activation layer for convolution 2			|
| Convolution 7x7       | 1x1 stride, Valid padding, outputs 8x8x64 	|
| RELU					| Activation layer for convolution 3			| 
| Fully connected		| 2 hidden layers (520 and 180)					|            
| Softmax				| Output layer 43 classes.						|
| Cross Entropy Loss	| Calculate loss of the model with one-hot encoded labels|
| Dropout				| Dropout on Fully connected layer 1 and 2 with keep probability 0.5|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer (similar to Stochastic Gradient Descent) with learning rate 0.001, number of epochs as 10, drop out probability as 0.5 and shuffled the training data every epoch. Also I have been calculating accuracy of the validation data after every epoch.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.00
* validation set accuracy of 0.96
* test set accuracy of 0.98

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? : 
Answer: I choose the Lenet architecture, but its validation accuracy was not upto 0.93 and then my observation found that max pooling layers accumulating in critical data loss, so I decided to adjust the convolution layer without max pooling. Also, adding drop out in the fully connected hidden layer 1 and 2 improved the validation accuracy.

* What were some problems with the initial architecture?:
Answer: Validation accuracy was not upto 0.93 and then my observation found that max pooling layers accumulating in critical data loss.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
Answer: Removed pooling as  LeNet convolution network was underfitting, added 3 convolution layers to detect edges, complex shapes and then more complex shapes. Also added dropout to make sure that model is not overfitted.

* Which parameters were tuned? How were they adjusted and why?
Answer: 
1) Convolution layer filter parameters adjusted to gather maximum information and to make sure that accuracy is good.
2) Hidden layer 1 and 2 parameters were adjusted to make sure that model is not underfitted.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Convolution layer without pooling helped that data information is not lost and choosing dropout layer helped that model is not underfitted.
If a well known architecture was chosen:
* What architecture was chosen? 3 layers Convolution and 2 Hidden layers Fully Connected Network with Dropout.
* Why did you believe it would be relevant to the traffic sign application? Every convolution layer helped in detecting edges, complex shapes and then numbers and text to identify the traffic signs.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 Training, Validation and Test model accuracy is almost similar and that is recommending that model is working well.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt tag] (https://github.com/ankit2grover/CarND-Traffic-Sign-Classifier-Project/blob/master/DataSet/Web/german/2_1.png)

![alt tag] (https://github.com/ankit2grover/CarND-Traffic-Sign-Classifier-Project/blob/master/DataSet/Web/german/11_1.png)

![alt tag] (https://github.com/ankit2grover/CarND-Traffic-Sign-Classifier-Project/blob/master/DataSet/Web/german/14_1.png)

![alt tag] (https://github.com/ankit2grover/CarND-Traffic-Sign-Classifier-Project/blob/master/DataSet/Web/german/25_1.png)

![alt tag] (https://github.com/ankit2grover/CarND-Traffic-Sign-Classifier-Project/blob/master/DataSet/Web/german/28_1.png)

My Convolution Network not able to classify traffic speed numbers, it has low precision on them because it is not able to detect numbers really well.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit 50 		| Speed Limit 60   								| 
| Right of way at next	| Right of way at next 							|
| Stop					| Stop											|
| Road work	      		| Road work 					 				|
| Children crossing		| Children crossing 							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 90%. This compares favorably to the accuracy on the test set of accuracy.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 23rd cell of the Ipython notebook.

For the first image, the model predicted incorrent that this is a Speed limit (60km/h) (probability of 0.72), and the image does contain a Speed limit (50km/h). The top five soft max probabilities were

| Probability(%)       	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.72         			| Speed limit (60km/h)   									| 
| 0.10     				| Roundabout mandatory 										|
| 0.04					| Road work											|
| 0.03	      			| Ahead only					 				|
| 0.02				    | Bicycles crossing							|


For the first image, the model is relatively sure that this is a Right-of-way at the next intersection (probability of 1), and the image does contain a Right-of-way at the next. The top five soft max probabilities were

| Probability(%)       	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00      			| Right-of-way at the next   									| 
| 0.00   				| Beware of ice/snow 										|
| 0.00					| Double curve											|
| 0.00	      			| Pedestrians					 				|
| 0.00				    | Priority road							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

You can see the input image with a sign of Speed Limit(20 km/h) and convolution layer 1 with feature map of 16 is detecting edges of the image.

![alt tag] (https://github.com/ankit2grover/CarND-Traffic-Sign-Classifier-Project/blob/master/OutputImages/convlayer1ouput.png)






