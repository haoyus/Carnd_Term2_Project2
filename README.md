#**Traffic Sign Recognition Project Write-up** 


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

[image1]: ./Train_Pic_Samples.png "Visualization"
[image2]: ./New_Images.png "Five new images"


## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

You're reading it! and here is a link to my [project code](https://github.com/haoyus/Carnd_Term2_Project2/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. A basic summary of the data set. In the code, the analysis is done using python.

I used python len function to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. An exploratory visualization of the dataset.

This chart of images visualizes all unique classes of traffic signs in the data set:

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

First of all, I decided **NOT** to convert the images to grayscale because I think color might play a role in classifying these traffic signs. Plus, I'd like to test the performance of LeNet based CNN on recognizing colored images.

The only pre-processing that I did to the data set is that I normalized the image data so that the data has zero mean and equal variance.
The normalization is done by simply applying the following process:
```python
X_train = (X_train-128.)/128.
X_valid = (X_valid-128.)/128.
X_test = (X_test-128.)/128.
```

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is based on LeNet architecture, with a few modifications including change of fully connected layer output size and dropout. The final model can be summarized with the following table:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Layer 1: Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|					outputs 28x28x6				|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6 	|
| Layer 2: Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16		|
| RELU		| outputs 10x10x16			|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16 	|
| Flatten	      	| input 5x5x16, output 5x5x16=400 	|
| Layer 3: Fully Connected   | input 400, output 300		|
| RELU					|					outputs 300			|
| Dropout				|		Only applied for training,	outputs 300			|
| Layer 4: Fully Connected   | input 300, output 172		|
| RELU					|					outputs 172			|
| Dropout				|		Only applied for training, outputs 172			|
| Layer 5: Fully Connected   | input 172, Output size is n_classes=43		|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used 20 EPOCHS. Initially I used 10, but later on I figured 20 would train the model better.

The batch size I used is 128, which follows the LeNet original setting. Same for the training rate, I kepted it as 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image2]

I found these images by googling "German traffic signs", and seleted the ones that can be found in the data set.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 



