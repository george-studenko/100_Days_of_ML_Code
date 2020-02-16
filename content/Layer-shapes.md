## Layer Shapes Calculation in CNNs

Calculating the shape of layers have always been a hard thing for me, today I found this:

For any convolutional layer, the output feature maps will have the specified depth (a depth of 10 for 10 filters in a convolutional layer) and the dimensions of the produced feature maps (width/height) can be computed as the input image width/height, W, minus the filter size, F, divided by the stride, S, all + 1. 
The equation looks like: ```output_dim = (Width - Filter + 2Padding) / Stride + 1```.

Where:
 ```Filter```: can be also referred as the ```Kernel```
```Padding```: Could be zero, in that case ```2 x 0 = 0```  which will simplify the equation to   ```output_dim = (Width - Filter) / Stride + 1```.

For a pool layer with a size 2 and stride 2, the output dimension will be reduced by a factor of 2. Read the comments in the code below to see the output size for each layer.

So for example for an input of 28 x 28 pixels in grayscale (1,28,28) 

Where 1 is the channels (since it is grayscale it is only 1) when applying one convolutional layer like this: ```nn.Conv2d(1, 10, 3)``` 

Where ```nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)``` so 1 channel in, 10 out and a kernel of 3 x 3 then the output layer will have a shape of (10,26,26) ```10``` because that is what we specified as output channel and 26 because it is the result of the formula ```output_dim = (W-F)/S + 1``` that being translated into ```=((28-3)/1)+1``` (28 pixels - 3 of the kernel (or filter) / by the stride which we didn't specify so by default that is 1 and then + 1) = 26

If then we apply a MaxPooling layer with a kernel of 2 x 2 and stride of 2 x 2 then the formula is easier just divide it by 2! 

So if we apply this ```nn.MaxPool2d(2, 2)``` then we just need to divide the previous result ```26``` that is ```26 / 2 = 13``` that will give us a layer of shape ```(10,13,13)``` 10 since we keep the same number of outputs so we still take 10 as number of inputs.

If at any moment the formula yields a result with decimals, the number will be rounded down (just get rid of the decimal part) 
