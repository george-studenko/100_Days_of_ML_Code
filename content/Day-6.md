# Defining a CNN Network architecture
Convolutional Neural networks will make use of the following types of layers:
* Convolutional Layers
* Maxpooling layers
* Fully connected (linear) layers 

To define a network in PyTorch you create a class:

**class Net**(nn.Module)

Define the layers on the __init__ function

```def __init__(self, n_classes):
  super(Net,self).__init__()
  # 1 input channel (grayscale image), 32 outputs or feature maps, 5x5 conv. kernel
  self.conv1 = nn.Conv2d(1,32,5) 
  
  # pool with kernel_size = 2, stride = 2
  self.pool = nn.MaxPool2d(2,2)
  
  # fully connected layer,  input_size, number of output classes
  self.fc1 = nn.Linear(32*4,n_classes)
```

The feedforward behaviour is defined in the **forward** function which takes the image tensor x as input:

```
def forward(self,x):
   # one conv/relu + pool layers
   x = self.pool(F.relu(self.conv1(x)))

   # prep for linear layer by flattening the feature maps into feature vectors
   x = x.view(x.size(0), -1)
   # linear layer 
   x = F.relu(self.fc1(x))
   
   # final output
   return x
```