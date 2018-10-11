# 100 Days of Machine Learning Code

100 Days of Machine Learning Code. Full instructions about the challenge [here](https://github.com/george-studenko/100_Days_of_ML_Code/blob/master/100_Days_of_ML_Code_Instructions.MD)

Here you can find a very useful [Machine Learning Glossary](https://developers.google.com/machine-learning/glossary/)

## Day 12 : Oct 11, 2018

**Today's Progress** : Kept working on the Facial Keypoints Project, I changed the code to make it run on my local GPU, and that did help a lot, I could test many different architectures very quick, and now I finally have a decent network. Was also working on show the Feature maps visualizations, by applying the kernels that the network learned during training to the images we can see exactly what the network sees, check the notes for a screenshot

**Thoughts** : If you can run it on a GPU, do it, it is just hundreds of times faster, your time is worth it. it is really cool to see the feature maps that the network created and used to learn!

**Link to work:** [Day 12 notes](content/Day-12.md) | [Filtered Images](resources/kernels.png)

## Day 11 : Oct 10, 2018

**Today's Progress** : Kept trying different architectures on the Facial Keypoints Project, Loss functions, and optimizers, I changed my input images from 96x96 to 224x224 and the training time increased by 5 or 10 at least, also the loss increased so it is performing worse than before. 

**Thoughts** : I got to keep working on it, will try to make it run on GPU to be able to experiment more.

## Day 10 : Oct 9, 2018

**Today's Progress** : Working on the Facial Keypoints Project, reading a paper about Naimishnet to have an idea of the implemented architecture, implemented my architecture although I'm still experimenting with it.

**Thoughts** : As always there is a lot of experimentation when it comes to the architecture and hyperparameters tunning, so I'm still testing different sets of layers and activation functions, loss functions and other hyperparameters.

**Link to work:** [Day 10 notes](content/Day-10.md) | [Architecture](code/cv/models.py)

## Day 9 : Oct 8, 2018

**Today's Progress** : I've been working on the CV nanodegree, learning more about ways to improve a network with Dropout and Momentum and also how to decide what layers to use when developing a network (hint: it is all about intuitions).

**Thoughts** : Momentum seems to be a good technique to use when local minimas might be a problem, and dropout will help a lot to make all nodes of a network better and avoid having some nodes that might be doing all the job opn their own. While deciding on what layers to include is all about intuitions there are some things that you can try out to see if the model improves check the list in the notes of day 9.

**Link to work:** [Day 9 notes](content/Day-9.md)

## Day 8 : Oct 7, 2018

**Today's Progress** : I've been working on the CV nanodegree, although I didn't advance much I took time to really understand how the shape of layers change after each convolutional layer and after each pooling layer, since that is one of the main confussions I had with CNNs for a long time.

**Thoughts** : it is pretty simple to understand once you do it a couple of times, I have also been testing CNNs architectures adding more layers and finishing with Convolutions or pooling layers before the last fully connected layer and when trying that I figured out that when having a Convolutional layer before the fully connected the accuracy was of 4.6% but if instead I used a max pooling layer before the FC layer the accuracy went up to 10%.

**Link to work:** [Day 8 notes](content/Day-8.md)

## Day 7 : Oct 6, 2018

**Today's Progress** : Working on the Fast.AI Deeplearning course on data augmentation and ways to find the best learning rate

**Thoughts** : It's interesting to realize that data augmentation is not always benefitial and to understan when it is, also it is interesting to see that are some crucial concepts like the lr_find function that is based on the part of a not really known paper, that is just great to find the optimal learning rate.

**Link to work:** [Day 7 notes](content/Day-7.md)

## Day 6 : Oct 5, 2018

**Today's Progress** : learning about CNNs types of layers and how to define the network and feedforward with PyTorch

**Thoughts** : Still a bit confused when it comes to define the actual size (shape) of the network after a couple of layers.

**Link to work:** [Day 6 notes](content/Day-6.md)

## Day 5 : Oct 4, 2018

**Today's Progress** : Progress on the Computer Vision ND, learning about ORB and HOG algorithms, and starting with Convolutional Neural Networks

**Thoughts** : ORB is easier to understand than HOG, both serve a different purpose, I've some previous experience with CNNs but I was struggling with it, this time I'm getting a better grab of it.

**Link to work:** [Day 5 notes](content/Day-5.md)

## Day 4 : Oct 3, 2018

**Today's Progress** : I decided to go a bit less technical today and research about computer vision jobs, it is a very interesting field of study and it can be applied to so many things. 

**Thoughts** : From medicine to self-driving cars, it is just amazing and inspiring at the same time the things that can be accomplished with computer vision.

**Link to work:** [Jobs in Computer Vision](content/Jobs-in-Computer-Vision.md)

## Day 3 : Oct 2, 2018

**Today's Progress** : I finished the Types of Features and Image Segmentation material of the Computer Vision Nanodegree and going back to Tensorflow after many months with the Google ML Crash course

**Thoughts** : Looks like detecting corners is a quite useful technque in Computer Vision, also image contouring seems to be a great way to detect images. K-means clustering can group part of the image together to help with image segmentation! I have learned about high features in Tensorflow, I'm not sure if those are new features or I just didn't get the chance to use them before.

**Link to work:** [Day 3 Notes](content/Day-3.md) | [Contour detection and features](https://github.com/george-studenko/100_Days_of_ML_Code/blob/master/code/cv/Contour%20detection%20and%20features.ipynb)


## Day 2 : Oct 1, 2018

**Today's Progress** : Finished the lessons: Image Representation and Classification, Convolutional Filters and Edge Detection from the computer Vision Nanodegree, only 3 lessons to go to start with the first project: Facial Keypoint Detection.

**Thoughts** : Edge detection can be quite useful to detect images, Hough transformations are also quite useful to detect lines that correspond to the same object, lots of parameters to adjust and tune. Bluring images before applying Canny edge detection is quite useful to get rid of unwanted noise in images.

**Link to work:** [Finding Edges and Custom Kernels](code/cv/Finding%20Edges%20and%20Custom%20Kernels.ipynb) | [Hough Lines](code/cv/Hough_lines.ipynb) | [Detecting edges live with webcam repo](https://github.com/george-studenko/Live-Sketch-with-Computer-Vision)


## Day 1 : Sep 30 , 2018

**Today's Progress** : I've started to work on the Machine Learning Crash Course with TensorFlow APIs Google's fast-paced, practical introduction to machine learning today.
  
**Thoughts** : This is a good beginners course, useful to refresh some concepts and get to practice tensorflow.

**Link to course:** [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/ml-intro)

**Link to work:** [Day 1 Notes](content/Day-1.md) | [Intro to Pandas Lab](code/Intro_to_pandas.ipynb)


## Day 0 : Sep 29 , 2018
 
**Today's Progress** : I have joined and attended [AI Saturdays](https://nurture.ai/ai-saturdays) in Barcelona today, I completed a couple of chapters on the Computer Vision Nanodegree I'm working on right now. I've been working on creating masks to do some image edition with CV2 (OpenCV). Learned about color spaces (RBG, HSI and HSL) and worked on a simple image classifier, creating a basic ML pipeline to load images, preprocess images and predict and image label.  

**Thoughts** : Getting to work and know people who is interested in AI, ML and DL is great to share knowledge and encourage each other, a great way to learn. Computer vision is a daunting part of AI, but with the right material, teachers and projects I think I will manage to pull it over, got time until the end of December 2018 to get it done.

**Link to course:** [Computer Vision Nanodegree](https://eu.udacity.com/course/computer-vision-nanodegree--nd891) you can find a [Free Preview here](https://www2.udacity.com/course/ud891-preview)
