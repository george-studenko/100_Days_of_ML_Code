## Momentum
In gradient descent momentum uses a constant \beta between 0 and 1, and it is used to calculate the next step size, it will weight previous steps, so the previous step matthers a lot and the weight for each previous step will then decrease and this is done by using the the constant \beta so: 
* the previos step will be multiplied by 1 
* the step before that \times \beta 
* the one before that \times \beta<sup>2</sup> 
* the one before that \times \beta<sup>3</sup>
* and so on...

Momentum is specified in the optimizer, example:

```optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)```

## Network structure
The hardest part if to develop an intuition to decide on what layers you will include in a given network since the network will perform different depending on the complexity of the task, some recomended things to try are:

* Change the number of convolutional layers 
* Increase the size of convolutional kernels for larger images
* Change loss and optimization functions 
* Change hyperparameters such as learning rate and momentum
* Add layers to prevent overfitting
* Change the batch_size of the data loader to see how larger batch sizes can affect training

Make notes on how the loss changes and tune it from there.