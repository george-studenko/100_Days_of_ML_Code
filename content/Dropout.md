## Dropout 
Will randomly turn on and off nodes in each layer (with some specified probability) each epoch during the feedforward and backpropagation, that means that the node that was disabled will not contribute to the prediction and will neither get the weighst updated during backpropagation, this will help the model generalize better and increase accuracy on the test dataset. By turning off nodes randomly it makes all nodes work better as a team by making sure no node is too weak or too strong. 

Dropout layers often come near the end of the network; placing them in between fully-connected layers for example can prevent any node in those layers from being too strong.

Dropout is specified as part of the network inside the ```__init__``` method, example:
```self.dropout = nn.Dropout(p=0.2)```

Dropout will be defined with a probability of a node being dropped, so when ```ps=0.25``` there is a 25% probability of that node to be dropped.

The bigger the network and the more parameters a network is more likely to overfit, to fix that having a higher dropout probability might help to avoid overfitting.

Some standard dropout probs are 0.25 for the first layer, 0.5 for a second layer and they are placed before linear fully connected layers.

In practice I've seen up to 6 dropout layers increasing on each layer from 0.1 probability to 0.6 for the last dropout layer.