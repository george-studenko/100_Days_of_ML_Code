# 100 Days of Machine Learning Code

100 Days of Machine Learning Code. Full instructions about the challenge [here](https://github.com/george-studenko/100_Days_of_ML_Code/blob/master/100_Days_of_ML_Code_Instructions.MD)

Here you can find a very useful [Machine Learning Glossary](https://developers.google.com/machine-learning/glossary/)

## Day 27 : Oct 26, 2018

**Today's Progress** : Was reading the paper [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/pdf/1411.4555.pdf) to understand better the project I'm working on. I was also working on the decoder network.

**Thoughts**: Still a bit lost with this project, it is not going to be easy, but nothing is when it comes to deep learning! 

**Link to work:** [Show and Tell: A Neural Image Caption Generator Paper](https://arxiv.org/pdf/1411.4555.pdf)



## Day 26 : Oct 25, 2018

**Today's Progress** : Kept working on the Image Captioning project, starting to define the CNN and RNN models that I will start using, for the CNN encoder I will be using the pre-trained ResNet-50 as for the RNN I'm still working on it but I will be using LSTMs cells. I also found an insteresting Pandas Dataframes cheatsheet that I'm adding to the collection, check the link below.

**Thoughts**: I need more practice with RNNs, I feel more confident with CNNs at the moment but I should be able to pull it out.

**Link to work:** [Pandas DataFrame Cheatsheet](cheatsheets/Pandas-DataFrame-Notes.pdf)

## Day 25 : Oct 24, 2018

**Today's Progress** : Working on the Image Captioning project that uses the [Microsoft COCO Dataset](http://cocodataset.org/#home), pre-processing images tokenizing captions.

**Thoughts**: I'm just starting on the project, it is the first time I will use different architectures on the same project and feeding a network the output of the previous network, will be an interesting project to do.

**Link to work:** [Image Captioning Project Notes](content/Image-Captioning.md)

## Day 24 : Oct 23, 2018

**Today's Progress** : Learning about Image Captioning and Tokenization with NLTK, exploring the [Microsoft COCO Dataset](http://cocodataset.org/#home) and starting to work on the Image Captioning project.

**Thoughts**: NLTK is an interesting package that has some extra functionalities to tokenize words, sentences or even tweets. The Image captioning project seems quite interesting and will help me to learn  more about RNNs, LSTMs and CNNs, and not only that but to combine different networks to get even more interesting stuff.

**Link to work:** [NLTK](content/NLTK.md)


## Day 23 : Oct 22, 2018

**Today's Progress** : Learning about more about attention mechanisms, a way in which the neural network focuses its attention on the most relevant parts of the input data. Now studying about Image Captioning.

**Thoughts**: Attention mechanisms are quite interesting, shifting attention to different parts of an image or different words in a text is something I wouldn't have expected to be able to do with a neural network

**Link to work:** [Attention mechanisms](content/Attention-mechanisms.md)


## Day 22 : Oct 21, 2018

**Today's Progress** : Learning about embeddings and starting to learn about attention mechanisms 

**Thoughts**: Embeddings are a bit hard to understand at the beginning, but they are just some hidden units that represent a categorical value that is translated to a vector of continuous values and are considered just another parameter of the network with their own weights so they will be trained and will take part of back-propagation as all other nodes in the network.

**Link to work:** [Embeddings](content/Embeddings.md)


## Day 21 : Oct 20, 2018

**Today's Progress** : Learning about Model Hyper parameters, a bit more on dropout and continued learning about embeddings.

**Thoughts**: It is good to have a rough idea on starting values for model hyperparameters, got to read more about embeddings while I've been using embeddings in the past I didn't really understand what they are until now, but still would like to get into more details to understand it better.

**Link to work:** [Hyperparameters](content/Hyperparameters.md) | [Dropout](content/Dropout.md)

## Day 20 : Oct 19, 2018

**Today's Progress** : Yet more on LSTMs, added code examples in the LSTM notes file, and back to basics on hyperparameters, today I studied about Optimizer Hyperparameters, next I will also add information about Model Hyperparameters

**Thoughts**: Hyperparameters are all about intuition and experience, there is no one fit for all so you will have to experiment most of the time, however you will be able find the best hyperparameters quicker and quicker over time!

**Link to work:** [Hyperparameters](content/Hyperparameters.md) | [LSTM cells](content/LSTM.md) 

## Day 19 : Oct 18, 2018

**Today's Progress** : Day four of the Fast.AI marathon working on lesson 4, still in progress!, continued learning about LSTMs and how to implement basic LSTM cells and get the outputs and hidden state with PyTorch.

**Thoughts**: LSTMs are a bit complex to understand at first, I need a bit more of practice with them to really understand them, I already understand the concept and how the cell works so that is a good start. 

**Link to work:** [LSTM cells](content/LSTM.md) 

## Day 18 : Oct 17, 2018

**Today's Progress** : Day three of the Fast.AI marathon completed the video of lesson 3!, interesting things on different activation functions, like softmax for non-binary classifications and sigmoid for multiple classes classification (predicting multiple correct outputs), also checking where and how to get the datasets. Also finally got the Kaggle Fastai kernels to work! Also learning more about what is inside an LSTM cell which in short are 4 gates: Learn, Forget, Remember and Use gates.

**Thoughts**: It is interesting to lean that activation functions have their own "personality" 

**Link to work:** [LSTM cells](content/LSTM.md) | [Downloading Datasets](content/Downloading-Datasets.md) | [FastAI](content/fast-ai.md) 


## Day 17 : Oct 16, 2018

**Today's Progress** : Day two of the Fast.AI marathon completed the video of lesson 2!, there are some really cool stuff in this lesson, unfortunately for people just starting on deep learning it won't be much appreciated (if this was for real their 2nd lesson on DL) like for example some neat tricks to find great Learning Rates like the Cosine Annealing with warm restarts and cycles. I then kept working on the computer vision program, I've been learning about more about Recurrent Neural Networks (RNNs) Backpropagation Through Time, and also Long Short-Term Memory cells (LSTMs)

**Thoughts**: BPTT is quite complex, too many terms to work with, but well it is just backpropagation but with many more terms involved. I'm learning a lot of neat tricks with Jeremy @ fastai, and now starting to understand all the complexity below the LSTMs cells.  

**Link to work:** [Learning Rates](content/Learning-rate.md) | [RNNs](content/RNN.md) 

## Day 16 : Oct 15, 2018

**Today's Progress** : Day one of the Fast.AI marathon completed the video of lesson 1!, and also worked on the computer vision program, I've been learning about Recurrent Neural Networks (RNNs), how the folded and unfolded models are represented, back to basics, Feedforward and Backpropagation and started learning about Backpropagation Through Time (BPTT)

**Thoughts**: Recurrent neural networks will add memory to a neural network, so each time we feed an input we will also feed the activation of the previous hidden layer.

**Link to work:** [RNNs](content/RNN.md)

## Day 15 : Oct 14, 2018

**Today's Progress** : I've been learning about YOLO (You Only Look Once) algorithm, I've also been documenting how to troubleshoot Conda environments problems with Jupyter notebooks and trying to make OpenCV work properly in Ubuntu.

**Thoughts**: YOLO seems to be amazing, you do need a GPU for real time recognition, with CPU it can take 2 to 3 seconds to analyze an image and return the outputs, looking forward to start using it!

**Link to work:** [YOLO](content/YOLO.md) | [Conda](content/Conda.md) 


## Day 14 : Oct 13, 2018

**Today's Progress** : Back to basics day!, on training, dev and testing sets, recognizing and fixing high variance and high bias problems and a bit of regularization and weight decay.  

**Thoughts** : Having a clear idea of how to split your data is important. Knowing what to do when you have problems of high bias and/or high variance is important, there are suggested steps to follow to fix those! 

**Link to work:** [Training Dev and Testing sets](content/Training-Dev-and-Testing-Sets.md) | [High bias & high variance](content/High-bias-and-high-variance.md) | [Regularization](content/Regularization.md)

## Day 13 : Oct 12, 2018

**Today's Progress** : I submitted the Facial Keypoints Project and it was approved, now stating to learn about more advanced features to work with multiple objects in a scene, with algorithms like R-CNNs  

**Thoughts** : I had lots of challenging moments while working on the project which were great to learn, I will try to implement more things on the notebook 4 of the project.

**Link to work:** [Day 13 notes](content/Day-13.md) | [Facial Keypoints Detector Project](https://github.com/george-studenko/Facia-Keypoints-Detector)


## Day 12 : Oct 11, 2018

**Today's Progress** : Kept working on the Facial Keypoints Project, I changed the code to make it run on my local GPU, and that did help a lot, I could test many different architectures very quick, and now I finally have a decent network. Was also working on show the Feature maps visualizations, by applying the kernels that the network learned during training to the images we can see exactly what the network sees, check the notes for a screen shot

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

**Link to work:** [Day 9 notes](content/Day-9.md) | [Dropout](content/Dropout.md)

## Day 8 : Oct 7, 2018

**Today's Progress** : I've been working on the CV nanodegree, although I didn't advance much I took time to really understand how the shape of layers change after each convolutional layer and after each pooling layer, since that is one of the main confusions I had with CNNs for a long time.

**Thoughts** : it is pretty simple to understand once you do it a couple of times, I have also been testing CNNs architectures adding more layers and finishing with Convolutions or pooling layers before the last fully connected layer and when trying that I figured out that when having a Convolutional layer before the fully connected the accuracy was of 4.6% but if instead I used a max pooling layer before the FC layer the accuracy went up to 10%.

**Link to work:** [Day 8 notes](content/Day-8.md)

## Day 7 : Oct 6, 2018

**Today's Progress** : Working on the Fast.AI Deeplearning course on data augmentation and ways to find the best learning rate

**Thoughts** : It's interesting to realize that data augmentation is not always beneficial and to understan when it is, also it is interesting to see that are some crucial concepts like the lr_find function that is based on the part of a not really known paper, that is just great to find the optimal learning rate.

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

**Thoughts** : Looks like detecting corners is a quite useful technique in Computer Vision, also image contouring seems to be a great way to detect images. K-means clustering can group part of the image together to help with image segmentation! I have learned about high features in Tensorflow, I'm not sure if those are new features or I just didn't get the chance to use them before.

**Link to work:** [Day 3 Notes](content/Day-3.md) | [Contour detection and features](https://github.com/george-studenko/100_Days_of_ML_Code/blob/master/code/cv/Contour%20detection%20and%20features.ipynb)


## Day 2 : Oct 1, 2018
0
**Today's Progress** : Finished the lessons: Image Representation and Classification, Convolutional Filters and Edge Detection from the computer Vision Nanodegree, only 3 lessons to go to start with the first project: Facial Keypoint Detection.

**Thoughts** : Edge detection can be quite useful to detect images, Hough transformations are also quite useful to detect lines that correspond to the same object, lots of parameters to adjust and tune. Bluring images before applying Canny edge detection is quite useful to get rid of unwanted noise in images.

**Link to work:** [Finding Edges and Custom Kernels](code/cv/Finding%20Edges%20and%20Custom%20Kernels.ipynb) | [Hough Lines](code/cv/Hough_lines.ipynb) | [Detecting edges live with webcam repo](https://github.com/george-studenko/Live-Sketch-with-Computer-Vision)


## Day 1 : Sep 30 , 2018

**Today's Progress** : I've started to work on the Machine Learning Crash Course with TensorFlow APIs Google's fast-paced, practical introduction to machine learning today.
  
**Thoughts** : This is a good beginners course, useful to refresh some concepts and get to practice tensorflow.

**Link to course:** [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/ml-intro)

**Link to work:** [Day 1 Notes](content/Day-1.md) | [Intro to Pandas Lab](code/Intro_to_pandas.ipynb)


## Day 0 : Sep 29 , 2018
 
**Today's Progress** : I have joined and attended [AI Saturdays](https://nurture.ai/ai-saturdays) in Barcelona today, I completed a couple of chapters on the Computer Vision Nanodegree I'm working on right now. I've been working on creating masks to do some image edition with CV2 (OpenCV). Learned about color spaces (RBG, HSI and HSL) and worked on a simple image classifier, creating a basic ML pipeline to load images, pre-process images and predict and image label.  

**Thoughts** : Getting to work and know people who is interested in AI, ML and DL is great to share knowledge and encourage each other, a great way to learn. Computer vision is a daunting part of AI, but with the right material, teachers and projects I think I will manage to pull it over, got time until the end of December 2018 to get it done.

**Link to course:** [Computer Vision Nanodegree](https://eu.udacity.com/course/computer-vision-nanodegree--nd891) you can find a [Free Preview here](https://www2.udacity.com/course/ud891-preview)
