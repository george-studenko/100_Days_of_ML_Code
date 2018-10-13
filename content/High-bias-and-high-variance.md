## Bias and Variance
When looking at the train set error and comparing to the cross validation error we could have the following situations:

Train set error: 1% (the model is doing very well)
Cross Val error: 11% 

There is a big variance in error between the training set and cross val. set, this is an example of **High Variance**, and it happens because the model cannot generalize well so it makes a lot of mistakes when using new data.

Train set error: 15% (the model is not doing very well)
Cross Val error: 16% (similar to the training set, not big variance)

This is an example of **High Bias**, the model needs to train more to get better

Train set error: 15% (the model is not doing very well)
Cross Val error: 30% (big variance)

This is an example of both previous cases together, the model is not performing well on the training set so that is already an example of high bias and the variance with the dev set is huge so that adds high variance as well. So in this example we have High Bias and High Variance.

Train set error: 1% (the model is doing very well)
Cross Val error: 2% (similar to the training set, not big variance)

This is a perfect example of **Low Bias** and **Low Variance**

Base error or human error is the error % a person will have when performing the same task

## Basic recipe for ML
* Does the model have high bias? (training data performance)
Try a bigger network, train longer (more epochs), try a different NN architecture, keep doing it until the training data fits well.

* Does the model have High Variance? (Dev Set performance) 
Add more data, data augmentation, try regularization, NN architecture.

For High variance add data
For High Bias improve the network


