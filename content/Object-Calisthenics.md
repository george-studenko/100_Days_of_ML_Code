# Object Calisthenics
By applying object calisthenics you can get code that is more:
* Readable  
* Reusable  
* Testeable
* Maintenable 

## Rules

### 1. Only **one** level of indentation per method.  
This will make a method more readable as you will need to extract out pieces of code for example for loops or conditionals and put a meaningful name to those methods which are going to be called in this method.  
#### Benefits  
* Single responsibility
* Better naming
* Shorter methods
* Reusable methods

### 2. Don't use the **else** keyword
Else keywords will add complexity to the code and create non linear flows (adding more use cases)

### Ways to avoid using the else keyword
* Use default values
* Early return
* Extract code
* Use polymorphism
* State pattern
* Strategy pattern

#### Benefits
* Avoid code duplication
* Lower complexity 
* Readability 

### 3. Wrap primitive types
By wrapping primitive types into classes we can encapsulate the type and have control from a single place in case we need to refactor or change the primitive type later on, it also helps to make more readable by giving a hint of what exactly a method parameter is receiving

#### Benefits
* Encapsulation
* Type hinting
* Attracts similar behavior

### 4. One dot per line
This applies the Demeter law: *Only talk to your friends* 

#### Benefits
* Encapsulation
* Open/Close Principle

### 5. Do **not** abbreviate
If you are abbreviating it is quite possible that the method is doing more than it should (violating the single responsibility principle), so think again.

#### Benefits
* Single Responsibility principle
* Avoid confusion
* Avoid code duplication
 
 ### 6. Keep classes small
 * 15 - 20 lines per method
 * 50 lines per class
 * 10 classes per package
 
 #### Benefits
 * Single Responsibility principle
 * Smaller modules
 * Coherent code
 
 ### 7. No more than 2 instance variables per class
 A single class should handle only one state and two at most, so having more than 2 instance vars might be violating SRP.
 
 #### Benefits
 * High cohesion
 * Encapsulation
 * Fewer dependencies
 
 ### 8. First class collections
 This is similar to rule #3 but applied to collections
 
 #### Benefits
 * Wrap primitive collections
 * Behavior or related collections have a home
 * Encapsulation
 
### 9. Do not use getters and setters
Don't make decisions outside of the class, let the class do it's job, follows the *Tell don't ask principle*

#### Benefits
* Open/close principle
