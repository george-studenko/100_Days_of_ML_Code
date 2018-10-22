## Attention Mechanisms
Allows the network to pay more attention to the most important parts, so in an image it will find the most important pixels or in language transltation it will know what words to focus on and the order it has to process them.

The inputs are passed in to an encoder, the encoder will generate the context (hidden states for all sequences) and will pass in the context to the decoder network and the decoder will output the sequence output.

In computer vision the encoder will be a Convolutional Neural Network that will produce the feature vectors, in the [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf) paper it will use one of the latest convolution layers in the CNN of size 14 x 14 x 512 (that is 14 x 14 pixels image x 512 features) , flatten each feature into a 1-dimentional (14 x 14 = 196 x 1) vertical vector and end up with a matrix of 512 features resized size 196 x 512 which becomes the context vector.

The context vector will be sent as the input of the decoder.

There are 2 types of attention mechanisms, Additive attention and Mutiplicative attention. Not too long ago a new type have been implementet, The transformer, which uses Self-attention in the encoder and then Self-attention and Encoder-Decoder attention on the Decoder.

Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
