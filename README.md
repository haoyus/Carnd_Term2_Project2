# **Traffic Sign Recognition Project Write-up** 


---

**Build a Traffic Sign Recognition Project (from the project requirements)**

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


## Rubric Points (detailed requirements for submission)
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

## **You're reading it! and here is a link to my [project code](https://github.com/haoyus/Carnd_Term2_Project2/blob/master/Traffic_Sign_Classifier.ipynb)**
And in the following sessions I will walk you through it step by step.

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

For dropout, I used 0.5 as keep_prob for training. But for validation and test, the keep_prob is 1.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.961
* test set accuracy of 0.943
If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

I improved the solution by an iterative approach.

In the beginning, I started with the LeNet architecture for the following reasons:
* It was designed for classifying images.
* It has been fully explored and proven to be accurate.
* Its convolutional layers can detect images regardless of where they are in the whole picture.

However, the original LeNet was to deal with grayscale images with size 32x32x1 and only 10 classes, whereas we have RGB images with size 32x32x3 and 43 classes. So I modified the LeNet so that it can be applied to our case: I changed the input of LeNet from 32x32 to 32x32x3, and changed the final output size from 10 to 43.
So the first architecture I tried was:
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
| Layer 3: Fully Connected   | input 400, output 120		|
| RELU					|					outputs 120			|
| Layer 4: Fully Connected   | input 120, output 84		|
| RELU					|					outputs 84			|
| Layer 5: Fully Connected   | input 84, Output size is n_classes=43		|

With this architecture, I was able to achieve:
* training set accuracy of 0.993
* validation set accuracy of 0.902
This was not good. So I didn't test it with test set.

I thought the output sizes of fully connected layers might be a little too small for a final output size of 43, because 43 is much bigger than 10. I'd like to try increasing the output sizes of fully connected layers so that it contains sufficient information to classify 43 types.
Thus I make the following changes (highlighted with **bold**):
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
| Layer 4: Fully Connected   | input 300, output 172		|
| RELU					|					outputs 172			|
| Layer 5: Fully Connected   | input 172, Output size is n_classes=43		|

And with this bigger network I was able to achieve:
* training set accuracy of 0.996
* validation set accuracy of 0.932
The performance was improved! But it was barely over 0.93 on the validation set accuracy. With test set? I didn't try, because I reckoned that it might not look good.

Then a method came to my mind: **Dropout**.
From the classes, I learned that Dropout is an efficient way to improved the accuracy of DNN, better than pooling. Max pooling was applied here, then why don't I try applying dropout? So I applied dropout between layer 3 and layer 4, and then between layer 4 and layer 5.
Now the architecture becomes this:

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

With this new architecture, I was able to achieve:
* training set accuracy of 0.998
* validation set accuracy of 0.961
The validation set accuracy is much higher than 0.93! Now I can try the test set. The test set accuracy turned out to be 0.943, higher than 0.93.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report.

Here are five German traffic signs that I found on the web:

![alt text][image2]

I found these images by googling "German traffic signs", and seleted the ones that can be found in the data set. It seems difficult to find really blurred images online.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road      		| Priority road   									| 
| Yield     			| Yield 										|
| Stop					| Stop											|
| General caution	      		| General caution					 				|
| Turn left ahead			| Turn left ahead      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.3%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is as follows:
```python
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    
    softmaxes = sess.run(tf.nn.softmax(logits), feed_dict={x: X_new, keep_prob: 1.0})
    
    values, indices = sess.run(tf.nn.top_k(softmaxes, k=5))
    
np.set_printoptions(precision=10)
print('\n' + str(values))
print('\n' + str(indices))
```
With 6 as the value of print precision for the probabilities, the top 5 softmax probabilities for each image with the sign type were displayed as:
```python
[[ 1.  0.  0.  0.  0.]
 [ 1.  0.  0.  0.  0.]
 [ 1.  0.  0.  0.  0.]
 [ 1.  0.  0.  0.  0.]
 [ 1.  0.  0.  0.  0.]]

[[12  0  1  2  3]
 [13  0  1  2  3]
 [14  0  1  2  3]
 [18  0  1  2  3]
 [34  0  1  2  3]]
```
This means the model is 100% sure of it's decision. I was surprised and confused. Because I thought there had be some kind of probability that the model wasn't so sure. So I set the print precision to 10, which means I will show accuracy to e-10. But still, the results didn't change. So the probability that the model had when predicting each sign:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.         			| Priority road   									| 
| 1.     				| Yield 										|
| 1.					| Stop											|
| 1.	      			| General caution						 				|
| 1.				    | Turn left ahead     							|
