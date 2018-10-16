## Recurrent Neural Networks (RNNs)
Will mantain states, the output depdends on the current input as well as previous ones.

This means that the input of the next step will include their own inputs + the output of the hidden layer of the previous step (the memory cell) and we will refer to it instead of ```h``` (for hidden layer) to ```s``` for State (that refers to the current state or memory)

The state will add a new matrix of weights Ws

In RNNs we will use Backpropagation Through Time (BPTT)

It is recommended to use the unfolded model to visualize the chain rule when it comes to calculate BPTT, as it will be necessary to accumulate gradients and the unfolded model will help visualize all paths that need to be calculated before acummulating them.

With only one step, we will need to update 3 Weight matrices, W<sub>y</sub>, W<sub>s</sub> and W<sub>x</sub>

The current state (S<sub>t</sub>) depends on the current input and previous states that will be activated 

S<sub>t</sub> = activation(X<sub>t</sub> * W<sub>x</sub> + S<sub>t-1</sub> * W<sub>s</sub>

The current output will be S<sub>t</sub> multiplied by the output weight matrix.

y<sub>t</sub> = S<sub>t</sub> * W<sub>y</sub>  
or with and activation  
y<sub>t</sub> = activation(S<sub>t</sub> * W<sub>y</sub>)

where ```activation``` is any activation function like Sigmoid, ReLu, etc.

We cannot accumulate more than 8 or 10 time steps of memory with this alone, because with BPTT we will suffer of the vanishing gradient problem, to avoid this issue we will need to use Long Short-Term Memory cells (LSTMs)
