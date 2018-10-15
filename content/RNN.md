## Recurrent Neural Networks (RNNs)
Will mantain states, the output depdends on the current input as well as previous ones.

This means that the input of the next step will include their own inputs + the output of the hidden layer of the previous step (the memory cell) and we will refer to it instead of ```h``` (for hidden layer) to ```s``` for State (that refers to the current state or memory)

The state will add a new matrix of weights Ws

In RNNs we will use Backpropagation Through Time (BPTT)

