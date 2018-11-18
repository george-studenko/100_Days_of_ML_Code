## Computer Vision

## Types of Features and Image Segmentation

**Types of features:** Edges, Corners and Blobs.  

**Corner Detector:** Intersection of 2 edges, can be calculated with Sobel operators (Sobel x and Sobel y) Gx and Gy (G for Gradient)    

**Dilation** (add pixels to the boundaries of an object) and **erosion** (removes pixels along object boundaries) can be combined to fill in gaps in the image or eliminate noise.   

Some combined operations are: **opening**, which is erosion followed by dilation, This is useful in noise reduction in which erosion first gets rid of noise (and shrinks the object) then dilation enlarges the object again.

**Closing** is the reverse combination of opening; it’s dilation followed by erosion, which is useful in closing small holes or dark areas within an object.  

**Image Contouring:** Allow us to get the area, perimeter, center and bounding rectangle of an image. It can be obtained with a binary thresholded image with black and white pixels (inverted so the background is black) in openCV you can use cv2.findContours method.  

**K-means Clustering**: Separates an image into segments by clustering data points that have similar traits. K-means is an unsupervised learning method.

## Computer vision algorithms 

**ORB Algorithm**: Oriented Fast and Rotated Brief, creates feature vectors from detected keypoints and is invariant to rotations, changes in illumination, and noise.

**HOG Algorithm**: Histogram of Oriented Gradients works by creating histograms of the distribution of gradient orientations in an image and then normalizing them in a very special way. This special normalization is what makes HOG so effective at detecting the edges of objects even in cases where the contrast is very low. These normalized histograms are put together into a feature vector, known as the HOG descriptor, that can be used to train a machine learning algorithm, such as a Support Vector Machine (SVM), to detect objects in images based on their boundaries (edges).

**Convolutional Neural Networks**: this is a oversimplified example of an CNN architecture:  
Input Image -> Convolutional Layers -> Pooling layers -> Feature maps -> Fully Connected (linear) layer -> Class scores -> Predicted class

Image Pre-processing: 

1. Load image
2. Trimming
3. Resizing
4. Sharpening
5. Extending
6. Negating
7. Gray scaling


