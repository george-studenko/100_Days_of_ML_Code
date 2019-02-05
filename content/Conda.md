## Conda
I've gone through a lot of trouble setting up my enviroments, most of the times the problem is that it wont show up on my Jupyter Notebooks as Kernels so I cannot access them so here is a quick couple of commands to run to create an enviroment and make it show up as a Kernel in Jupyter.

To create an enviroment with Python 3.6  
```conda create --name myEnviromentName python=3.6```   

To activate the enviroment  
```source activate myEnviromentName```   

To set the environment up as jupyter kernel  
```python -m ipykernel install --user --name myEnviromentName --display-name "Python (myEnviromentName)"```   

If an error comes up saying No Module name 'decorator'  
```pip install decorator```
