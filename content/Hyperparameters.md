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
 ```numbers of layers```  
 ```number of hidden units```   
 ```models specific hyperparameters```
 
 