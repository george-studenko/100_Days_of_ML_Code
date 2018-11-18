## Data augmentation
In order to improve models without adding new data, we can do data augmentation, this will work well for images that do make sense to make transformations on top of, so for example it is not a good idea to do it for text or numbers because text the orientation of characters in these cases are important. 

But for example for satellite images or other types of pictures then it does make sense to do it.

To do data augmentation we can do transformations like rotation, increasing and decreasing brightness, image cropping, flipping, image location, scaling, etc.