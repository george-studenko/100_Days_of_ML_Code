## Day 1 : Sep 30 , 2018
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

[Go Back](README.md)