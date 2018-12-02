# Fully- Convolutional Neural Networks

In fully convolutional networks the last fully connected layer (linear layer) gets replaced by a 1 x 1 convolution. This allows to preserve the spatial information and removes the constraint of the input size image, in CNNs the fully connected layer shape is different for every image size but not in this case.

A Fully-Convolutional Neural Network consists of an Encoder (the convolutions that we normally use in a CNN) followed by a 1 x 1 convolution and a decoder which is the opposite of the encoder, it is called a transposed convolution or deconvolution, but it is not the same a in the context of a transposed matrix. A transposed convolution is essentially a a reverse convolution where the forward and backward passes are swapped 

The goal of the encoder is to extract features from the image the goal of the decoder is to upscale the output of the encoder.

FCNNs can also skip layers to preserve information and segment images, all layers must be added by the end of the network to have all the information.
