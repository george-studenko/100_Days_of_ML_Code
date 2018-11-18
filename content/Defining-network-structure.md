## Defining a network structure
The hardest part if to develop an intuition to decide on what layers you will include in a given network since the network will perform different depending on the complexity of the task, some recommended things to try are:

* Change the number of convolutional layers 
* Increase the size of convolutional kernels for larger images
* Change loss and optimization functions 
* Change hyperparameters such as learning rate and momentum
* Add layers to prevent overfitting
* Change the batch_size of the data loader to see how larger batch sizes can affect training

Make notes on how the loss changes and tune it from there.