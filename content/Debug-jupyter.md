# Debug in Jupyter notebooks 
## Variable inspector
A quick way to inspect variables can be done with the jupyter extension ```Variable Inspector``` if a more line by line debug is required then you can either use pdb or what I'm using now is ```PixieDebugger``` that gives me a more traditional way to debug code

## Debug with PixieDebugger

Install PixieDust using the following pip command: 
```pip install pixiedust``` 

Import it: 
```import pixiedust```

TO use PixieDebugger for a specific cell, add:  
```%%pixie_debugger```
to the top of the cell and run it.