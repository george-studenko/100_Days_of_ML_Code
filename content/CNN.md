# Defining a CNN Network architecture
Convolutional Neural networks will make use of the following types of layers:
* Convolutional Layers
* Maxpooling layers
* Fully connected (linear) layers 

To define a network in PyTorch you create a class:

**class Net**(nn.Module)

Define the layers on the __init__ function

```
def __init__(self, n_classes):
  super(Net,self).__init__()
  # 1 input channel (grayscale image), 32 outputs or feature maps, 5x5 conv. kernel
  self.conv1 = nn.Conv2d(1,32,5) 
  
  # pool with kernel_size = 2, stride = 2
  self.pool = nn.MaxPool2d(2,2)
  
  # fully connected layer,  input_size, number of output classes
  self.fc1 = nn.Linear(32*4,n_classes)
```

The feedforward behavior is defined in the **forward** function which takes the image tensor x as input:

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

##  CNN's Glossary
* **CNNs:** Convolutional neural network. That is, a network which has at least one convolutional layer. A typical CNN also includes other types of layers, such as pooling layers and dense layers.
* **Convolution:** The process of applying a kernel (filter) to an image
* **Kernel / filter:** A matrix which is smaller than the input, used to transform the input into chunks
* **Padding:** Adding pixels of some value, usually 0, around the input image
* **Pooling:** The process of reducing the size of an image through downsampling.There are several types of pooling layers. For example, average pooling converts many values into a single value by taking the average. However, maxpooling is the most common.
* **Maxpooling:** A pooling process in which many values are converted into a single value by taking the maximum value from among them.
* **Stride:** the number of pixels to slide the kernel (filter) across the image.
* **Downsampling:** The act of reducing the size of an image