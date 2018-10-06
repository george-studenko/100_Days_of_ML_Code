## Data augmentation
In order to improve models without adding new data, we can do data augmentation, this will work well for images that do make sense to make transformations on top of, so for example it is not a good idea to do it for text or numbers because text the orientation of characters in these cases are important. 

But for example for satellite images or other types of pictures then it does make sense to do it.

To do data augmentation we can do transformations like rotation, increasing and decreasing brightness, image cropping, flipping, image location, scaling, etc.

## Finding the optimal Learning Rate

The Fast.AI framework implemented the [Cyclical Learning Rates for Training Nueral Networks paper](https://arxiv.org/pdf/1506.01186.pdf) learn.lr_find() which in simple words starts with a very small learning rate and starts doubling it until the loss get worse so then it stops. When you reach that step you can plot the results and see the best learning rate when the loss was still improving and before it overshoots.

I would like to implement my own lr_find function in the future.

 