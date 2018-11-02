## Motion in Computer Vision
Motion can be tracked with a 2D motion vector. A  vector has a direction and magnitude which will determine the direction and amount of movement between one frame and the next one.

First we will have to defive special points to track like for example intersections or corners once we localize those features in both frames we can track the motion vector. The magniture of the vector (how much it moved) can be found by the Pythagorean theorem:    
a<sup>2</sup> +  b<sup>2</sup>  = c<sup>2</sup> 

so the magnitude =  sqrt(a<sup>2</sup> + b<sup>2</sup>)

as for the direction we can calculate the angle:

angle = tan<sup>-1</sup>(b/a)

Knowing  those 2 things we can track an object