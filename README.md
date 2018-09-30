# 100 Days of Machine Learning Code - LOG

100 Days of Machine Learning Code. Full instructions about the challenge [here](https://github.com/george-studenko/100_Days_of_ML_Code/blob/master/100_Days_of_ML_Code_Instructions.MD)

Here you can find a very useful [Machine Learning Glossary](https://developers.google.com/machine-learning/glossary/)

## Day 0 : Sep 29 , 2018
 
**Today's Progress** : I have joined and attended [AI Saturdays](https://nurture.ai/ai-saturdays) in Barcelona today, I completed a couple of chapters on the Computer Vision Nanodegree I'm working on right now. I've been working on creating masks to do some image edition with CV2 (OpenCV). Learned about color spaces (RBG, HSI and HSL) and worked on a simple image classifier, creating a basic ML pipeline to load images, preprocess images and predict and image label.  

**Thoughts** : Getting to work and know people who is interested in AI, ML and DL is great to share knowledge and encourage each other, a great way to learn. Computer vision is a daunting part of AI, but with the right material, teachers and projects I think I will manage to pull it over, got time until the end of December 2018 to get it done.

**Link to work:** [Computer Vision Nanodegree](https://eu.udacity.com/course/computer-vision-nanodegree--nd891) you can find a [Free Preview here](https://www2.udacity.com/course/ud891-preview)

## Day 1 : Sep 30 , 2018

**Today's Progress** : I've started to work on the Machine Learning Crash Course with TensorFlow APIs Google's fast-paced, practical introduction to machine learning today.

### Key ML Terminology  

* **Feature**: features are the input variables we feed into a network, it can be as simple as a signle number or more complex as an image (which in reality is a vector of numbers, where each pixel is a feature)  
* **Label**: is the thing we are predicting, it is normally refered as ```y```  
* **Prediction**: or predicted value if the value we predict with a previously trained model for a given output and it is refered as ```y'```   

### Regression vs. classification:  
* A regression model predicts continuous values.  
* A classification model predicts discrete values.  

#### Linear Regression

 Is a method for finding the straight line or hyperplane that best fits a set of points.   
 
 Line formula:  
 y =  wx + b 
 
 Where:   
 
 w = Weights  
 x = Input features  
 b = Bias
 
 Some convenient loss functions for linear regression are:   
 **L<sub>2</sub> Loss** = (y - y')<sup>2</sup>  

**Mean Square Error**: is the average squared loss per example over the whole dataset. To calculate MSE, sum up all the squared losses for individual examples and then divide by the number of examples  

![MSE](resources/mse.png "MSE formula")

When training a model we want to minimize the loss as much as possible to make the model more accurate.
  
**Thoughts** : This is a good beginners course, useful to refresh some concepts and get to practice tensorflow.

**Link to work:** [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/ml-intro)
