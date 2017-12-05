# **Behavioral Cloning** 

## Writeup Submission


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia_model.png "Model Visualization"
[image2]: ./examples/recovery_right.png "Recovery Image"
[image3]: ./examples/recovery_left.png "Recovery Image"
[image4]: ./examples/recovery_bridge.png "Recovery Image"
[image5]: ./examples/normal.png "Normal Image"
[image6]: ./examples/flipped.png "Flipped Image"
[image7]: ./examples/left_view.png "Left Camera Image"
[image8]: ./examples/center_view.png "Center Camera Image"
[image9]: ./examples/right_view.png "Right Camera Image"
[image10]: ./video.mp4 "video"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

### Data Collection and Training

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.  The final model was trained on my workstation with a 1080ti GPU for 10 epochs. Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 64-73) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 58). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 87).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to implement a regression network to predict my steering angle for the autonomous simulation.  We used the mean squared error for the loss function to minimize the error between the stering angle that the network perdicts and the ground truth steering measurement.

My first step was to use a convolution neural network model similar to the Nvidia autonomous groups published netowrk model.  I thought this model might be appropriate since it is more powerful than the basic leNet network that I started with.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model using the basic leNet network had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it reflected the more complex model and then increased the data used for training the network.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track on my first run, but I improved the driving behavior in these cases cases by driving more laps and providing recovery driving from left to right and right to left.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

![alt text][image10]

#### 2. Final Model Architecture

The final model architecture (model.py lines 64-73) consisted of a convolution neural network with the following layers and layer sizes 


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:|  
| Input         		| 160x320x3 RGB image 						    |
| Normalize        		| 160x320x3 RGB image 						    |
| Cropping Layer   		| 90x295x3 RGB image 						    |
| Convolution Layer     | 5x5 kernel, 2x2 stride, 24 output channels 	|
| RELU					|												|
| Convolution Layer     | 5x5 kernel, 2x2 stride, 36 output channels 	|
| RELU					|												|
| Convolution Layer     | 5x5 kernel, 2x2 stride, 48 output channels 	|
| RELU					|												|
| Convolution Layer     | 3x3 kernel, 1x1 stride, 64 output channels	|
| RELU					|												|
| Convolution Layer    	| 3x3 kernel, 1x1 stride, 64 output channels	|
| RELU					| 												|
| Flatten Layer			| Flatten Data         							|
| Dense Layer     		| Fully connected layer							|
| Dense	Layer           | Fully connected layer	       					|
| Dense	Layer           | Fully connected layer							|
| Dense	Layer           | Fully connected layer 						|
|						|												|
 

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

I used the simulator to collect data by driving 3 laps, the first lap was concentrated on center lane driving, the second lap recovered from drifting off the road to both the left and right and the final lap was a mix of center driving and recovery. 

I collected left, center, and right camera images from the vehicles cockpit as show below.

![alt text][image7]
![alt text][image8]
![alt text][image9]


Recovering from the left side and right sides of the road back to center is depicted by the images below:

![alt text][image2]
![alt text][image3]
![alt text][image4]


To augment the data sat, I also flipped images and inverted steering angle to provide more training data for the neural network and to simulate a track with right hand turns, since the course was mostly left hand turns. For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]

After the collection process, I had 22842 number of data points. I then preprocessed this data by normalizing the data and cropping the image to focus the network on the road.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the validation loss decreasing for each epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.
