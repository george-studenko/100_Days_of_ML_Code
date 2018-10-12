## Weighted Loss Functions
For image classification and localization we need to use 2 loss functions on the same network!, to calculte the predicted class we would use categorical cross entropy but to find the bounding box (which is a regression problem) we need to use something like an SME, L1 Loss or smoot L1 Loss.

The way to do it is to use a weighted sum of classification and regression losses (ex. ```0.5*cross_entropy_loss + 0.5*L1_loss```); the result is a single error value with which we can do backpropagation. This does introduce a hyperparameter: the loss weights.

## Region Proposal Algorithms
R-CNN is a Region Convolutional Neural Network

It is used to produce a limited set of cropped regions to analyze ROIs (Regions of Interest) 

The R-CNN is the least sophisticated region-based architecture

The next one is the Fast R-CNN algorithm. Fast R-CNN is about 10 times as fast to train as an R-CNN because it only creates convolutional layers once for a given image and then performs further analysis on the layer

And then we have the Faster R-CNN which is the faster of all of the three, the main difference with Fast R-CNN is that it does a quick binary classification  on the RoIs which will classify if there is an object on the feature map or not before actually running the classification.