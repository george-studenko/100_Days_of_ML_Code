## Momentum
In gradient descent momentum uses a constant \beta between 0 and 1, and it is used to calculate the next step size, it will weight previous steps, so the previous step matthers a lot and the weight for each previous step will then decrease and this is done by using the the constant \beta so: 
* the previos step will be multiplied by 1 
* the step before that \times \beta 
* the one before that \times \beta<sup>2</sup> 
* the one before that \times \beta<sup>3</sup>
* and so on...

Momentum is specified in the optimizer, example:

```optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)```