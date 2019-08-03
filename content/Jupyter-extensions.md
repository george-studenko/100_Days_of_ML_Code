# Jupyter Notebook Extension

To install the packages required with conda use the following commands:

```
conda install -c conda-forge jupyter_contrib_nbextensions
conda install -c conda-forge jupyter_nbextensions_configurator
conda install -c conda-forge jupyterthemes 
```

Then run ```jupyter notebook```

Open the Extensions configurations panel ```http://localhost:8888/nbextensions```

Enable the extensions and enjoy them!

Some extensions to autoformat code require autopep8 and yapf 

```
conda install -c conda-forge yapf 
conda install -c conda-forge autopep8 
```

Official documentation [here](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/install.html) 

## With pip
You can do the same with pip
```
pip install jupyter_contrib_nbextensions  
pip install jupyter_nbextensions_configurator  
pip install jupyterthemes   
jupyter contrib nbextension install   
```
