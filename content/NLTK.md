## NLTK
Stands for Natural Language Toolkit.

Tokenization is just splitting sentences in a list of words.
 
### Word Tokenization with Python built in functions
```word = text.split()```

### Word Tokenization with NLTK 
```
from nltk.tokenize import word_tokenize
words = word_tokenize(text)
```

### Sencentes Tokenization with NLTK
```
from nltk.tokenize import sent_tokenize
words = sent_tokenize(text)
```

[NLTK Documentation](http://www.nltk.org/api/nltk.tokenize.html)