## LSTM Cells

LSTM Cells will replace hidden layers on a Recurrent Neural Network, and can be stacked so you can have multiple hidden layers, all of them being LSTM cells.

It is comprised of 4 gates with 2 inputs and 2 outputs:

**Learn Gate:** it takes the **Short term memory** and the **Event** and combines them with a ```than``` function and then ignores a part of it by multiplying it by an _ignore factor_ i<sub>t</sub>. To calculate the _ignore factor_ it combines de Shor term memory with the event multiplies it by the _Ignore Weights_ and activates using a ```sigmoid``` function and outputs the **new Short Term Memory**.

**Forget Gate:** it takes the **Long term memory** and multiplies it by a _forget factor_ f<sub>t</sub> to calculate the _forget factor_ it combines de Shor term memory with the event multiplies it by the _Forget Weights_ and activates using a ```sigmoid``` function.

**Remember Gate:** it takes the **Forget Gate** output and adds it to the **Learn Gate** outputs the **new Long Term Memory**.

**Use Gate**: or output gate, will take the **Forget Gate** and activate it with ```tanh``` then Take the **Shor Term Memory** and activate it with ```sigmoid``` and then multiplies them, and that is the output.

## LSTM in PyTorch

To define a LSTM:
```lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=n_layers)```

To initialize the hidden state:  
```h0 = torch.randn(1, 1, hidden_dim)```   
```c0 = torch.randn(1, 1, hidden_dim)```   

We will need to wrap everything in Variable, input is a tensor
```inputs = Variable(inputs)```  
```h0 = Variable(h0)```  
```c0 = Variable(c0)```  

Get the outputs and hidden state  
```out, hidden = lstm(inputs, (h0, c0))```  

For Natural Language 

## Basic LSTM Network
The first layer of a LSTM Network should always be an embedding layer  which will take the vocabulary dictionary size as the input.

Before we initialize the network we need to define the ```vocabulary``` that is simply a dictionary (with unique words) and each word will have a numerical index, to do so we can use the following function:

```
def add_word(word,dictionary):    
    if word not in dictionary:	
        dictionary[word] = len(dictionary) 
```

In this example we will use a vocabulary and a different dictionary for the outputs

```
training_data = [
    ("The cat ate the cheese".lower().split(), ["DET", "NN", "V", "DET", "NN"]),
    ("She read that book".lower().split(), ["NN", "V", "DET", "NN"]),
    ("The dog loves art".lower().split(), ["DET", "NN", "V", "NN"]),
    ("The elephant answers the phone".lower().split(), ["DET", "NN", "V", "DET", "NN"])
]

word2idx = {} 
tag2idx = {} 

for sent, tags in training_data:            
    # create a dictionary that maps words to indices
    for word in sent:
        add_word(word,word2idx)            
    # create a dictionary that maps tags to indices
    for tag in tags:
        add_word(tag,tag2idx)    
```

## Defining the network and feedforward function

```
class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):        
        super(LSTMTagger, self).__init__()
        
        self.hidden_dim = hidden_dim        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)        
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):        
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):        
        embeds = self.word_embeddings(sentence)        
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)        
        tag_outputs = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_outputs, dim=1)        
        return tag_scores
```

Apart from LSTMs there are also other types of architectures that also work well, like for exampla the **Gated Recurrent Unit** or **GRU**, the GRU combines the forget and learn gate into a update gate and returns only a "New Working Memory" instead of a long-term and short-term