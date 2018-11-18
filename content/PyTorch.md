## PyTorch
We can define our own PyTorch modules, to do so we need to inherit from ```nn.Module```

To have a fully functional PyTorch layer we can create a constructor and call the parent class constructor.  
```def __init__(self):
  super().__init__()
```

Then all we need to do is to define the forward function and return the results of a forward pass.

## PyTorch Squeeze
  Is used to delete one rank  



