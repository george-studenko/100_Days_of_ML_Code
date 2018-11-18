## Weighted Loss Functions
For image classification and localization we need to use 2 loss functions on the same network!, to calculate the predicted class we would use categorical cross entropy but to find the bounding box (which is a regression problem) we need to use something like an SME, L1 Loss or smooth L1 Loss.

The way to do it is to use a weighted sum of classification and regression losses (ex. ```0.5*cross_entropy_loss + 0.5*L1_loss```); the result is a single error value with which we can do backpropagation. This does introduce a hyperparameter: the loss weights.
