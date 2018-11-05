## Localization
Robot localization in essence is based in two main steps:  
```Sense``` and ```Move```  

It will start with a initial belief (or prior) of maximum confusion where the probability distribution will be uniform (flat, which means it has the same value everywhere)  
  
Then it will start cycling through sensor measurements (```Sense```) and movements (```Move```)

When the robot moves it loses information and when it senses it gains information.

If the robot will move enough steps without sensing (updating beliefs) the probability distribution will be uniform (max confusion state)

Entropy measures the amount of uncertainty.

   