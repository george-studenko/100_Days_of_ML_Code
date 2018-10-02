**Types of features:** Edges, Corners and Blobs.  

**Corner Dectector:** Intersection of 2 edgeds, can be calculated with Sobel operators (Sobel x and Sobel y) Gx and Gy (G for Gradient)    

**Dilation** (add pixels to the boundaries of an object) and **erosion** (removes pixels along object boundaries) can be combined to fill in gaps in the image or eliminate noise.   

Some combined operations are: **opening**, which is erosion followed by dilation, This is useful in noise reduction in which erosion first gets rid of noise (and shrinks the object) then dilation enlarges the object again.

**Closing** is the reverse combination of opening; it’s dilation followed by erosion, which is useful in closing small holes or dark areas within an object.  

**Image Contouring:** Allow us to get the area, perimeter, center and bounding rectangle of an image. It can be obtained with a binary thresholded image with black and white pixels (inverted so the background is black) in openCV you can use cv2.findContours method.  

**K-means Clustering**: Separates an image into segments by clustering data points that have similar traits. K-means is an unsupervised learning method.