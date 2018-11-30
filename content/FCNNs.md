# Fully- Convolutional Neural Networks

In fully convolutional networks the last fully connected layer (linear layer) gets replaced by a 1 x 1 convolution. This allows to preserve the spatial information and removes the constraint of the input size image, in CNNs the fully connected layer shape is different for every image size but not in this case.

A Fully-Convolutional Neural Network consists of an Encoder (the convolutions that we normally use in a CNN) followed by a 1 x 1 convolution and a decoder which is the opposite of the  

The goal of the encoder is to extract features from the image the goal of the decoder is to upscale the output of the encoder.