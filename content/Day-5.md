**ORB Algorithm**: Oriented Fast and Rotated Brief, creates feature vectors from detected keypoints and is invariant to rotations, changes in illumination, and noise.

**HOG Algorithm**: Histogram of Oriented Gradients works by creating histograms of the distribution of gradient orientations in an image and then normalizing them in a very special way. This special normalization is what makes HOG so effective at detecting the edges of objects even in cases where the contrast is very low. These normalized histograms are put together into a feature vector, known as the HOG descriptor, that can be used to train a machine learning algorithm, such as a Support Vector Machine (SVM), to detect objects in images based on their boundaries (edges).

**Convolutional Neural Networks**: this is a oversimplified example of an CNN architecture:  
Input Image -> Convolutional Layers -> Pooling layers -> Feature maps -> Fully Connectect (linear) layer -> Class scores -> Predicted class

Image Pre-processing: 

1. Load image
2. Trimming
3. Resizing
4. Sharpening
5. Extending
6. Negating
7. Grayscaling


