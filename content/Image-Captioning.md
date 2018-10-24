## Image Captioning Project
In this project I will train a network with the COCO Dataset (**C**ommon **O**bjects in **Co**ntext). This dataset contains images and a set of 5 different captions per image.

I will train a CNN-RNN model by feeding it with the image and captions so the network will learn to generate captions given an image. Once trained I will give the trained network an image and expect to get a caption that describes that image.

### Image transformations
As always the images will need some preprocessing:

* Resize to a smaller size for example 256 x 256 pixels
* Random crop to 224 x 224 standard size for most pre-trained models like Densenet or 299 x 299 for Inception (v3)
* Randomly flipping the image
* And the most important transformation convert it to an Image Tensor ```transforms.ToTensor()```
* If using a pre-trained model, normalize the images using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225] 

### Captions tokenization
I will use NLTK to tokenize the captions from the dataset.
First set the caption to lower case and then tokenize words.  
```nltk.tokenize.word_tokenize(caption.lower())```  

I will use 3 special tags: ```<start>``` to signal that is is the start of the caption```<end>``` for the end of the caption and ```<unk>``` for unknown words, these words will be words with very low relevance, words that do not appear that many times given a threshold.

These special tags will be at the beginning of the vocab dictionary and their corresponign indices will be 0 for ```start```, 1 for ```end``` and 2 for ```unk```.

So a simple phrase like ```This is a short text``` might be tokenized as follow:

```['<start>', 'This', 'is', 'a', 'short', 'text', '<end>']```

Then the words will be replaced for the corresponding indexes in the vocabulary, so it might become something like this:

```[0, 3, 98, 754, 3, 396, 1]```





