## Batch Size
Batch size refers to the number of training examples utilized in one step or iteration. 

One step or iteration is one step of gradiend decent (one update of weights and parameters)

The batch size can be either:

* The same number of the total number of samples which makes **one step = an epoch**, this is called **batch mode**

* A number greater than one but smaller than the total dataset samples, so for example for a dataset of 1000 images and a batch size of 100  then **10 steps  = 1 epoch** thi sis called **mini-batch mode**

* Exactly **one** in this case the gradient and the network parameters are updated after each sample this is called **stochastic mode**.
