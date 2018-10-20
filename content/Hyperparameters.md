## Hyperparameters
Is a variable that we need to set to a value before we can train a Neural Network. 

There are no magic numbers, it all depends on the architecture, data and problem to solve, etc.

### Optimizer Hyperparameters

 ```learning rate```  
 Is the most important hyperparameter of all, typical values are: 0.1, 0.01, 0.001, 0.0001, 0.00001 and so on.
 
 If the error diminishes very slowly then increase the learning rate.
 
 If the error bounces back from going smaller, then bigger and the smaller again, then decrease the learning rate.
 
 Another technique is called Learning Rate Decay which just consists in decreasing the learning rate every _x_ epochs.
 
 Adaptive Learning are algorithms that will decrease and increase the learning rate when needed.
 
 Some Adaptive Learning Optimizers are:
* [AdamOptimizer](https://pytorch.org/docs/stable/optim.html?highlight=adam#torch.optim.Adam)
* [AdagradOptimizer](https://pytorch.org/docs/stable/optim.html?highlight=adagrad#torch.optim.Adagrad)
 
 ```mini batch size```
 Small mini batches will have more noise and be slower
 Big mini batches could result in worse accuracy 
 
 Typical mini batch sizes are 1,2,4,8,16,23,64,128,256 with 32 being a good starting point.
 
 ```epochs```  
 Epochs are also known as number of iterations.
 
 To choose the best number of epochs we should check the Validation Error and train while the validation keeps decreasing.

 ### Model Hyperparameters
 ```numbers of hidden units```
 For the number of hidden units, we need to have enough hidden units to learn the function, so for a simple function the hidden layes will be less than for a complex one. 
 
 However if we add too many hidden units the network will have too much capacity that it will just memorize the training set and that will lead to overfitting.
 
 So if your model is overfitting then you could reduce the number of hidden units, or use regularization like L2 regularization or Dropout.
  
 ```number of layers```   
 As for the number of hidden layers, looks like 3 hidden layers works quite well, and while adding more can be benefitial but rarely helps much more, the exception being Convolutional Neural Networks. 
 
 ```models specific hyperparameters```
 For RNNs deciding wether to use a normal RNN cell, a LSTM cell or GRU cells, the number of layers to stack and the embeddings dimentions. 
 