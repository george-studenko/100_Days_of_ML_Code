## Learning rates
**Cosine Annealing**: uses cos/2 (so half of the cosine function) to decrease the learning rate as it trains. 

We can use Cosine Annealing on Stochastic Gradiend Descent with _warm restarts_, that means that when a cycle ends the learning rate will jump again to the highest learning rate (restart on top of the cosine function) and if the loss didn't increase it means we just landed on a good part of of the gradient which is quite flat.

I found this [artile](https://www.jeremyjordan.me/nn-learning-rate/) on learning rates which is quite interesting, there is a section talking about _Stochastic Gradient Descent with Warm Restarts (SGDR)_

## Finding the optimal Learning Rate

The Fast.AI framework implemented the [Cyclical Learning Rates for Training Nueral Networks paper](https://arxiv.org/pdf/1506.01186.pdf) learn.lr_find() which in simple words starts with a very small learning rate and starts doubling it until the loss get worse so then it stops. When you reach that step you can plot the results and see the best learning rate when the loss was still improving and before it overshoots.


 