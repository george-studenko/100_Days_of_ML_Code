# Feather

## What is Feather
Feather is a fast, lightweight, and easy-to-use binary file format for storing data frames. It has a few specific design goals:  

* Lightweight, minimal API: make pushing data frames in and out of memory as simple as possible
* Language agnostic: Feather files are the same whether written by Python or R code. Other languages can read and write Feather files, too.
* High read and write performance. When possible, Feather operations should be bound by local disk performance.

## How to install Feather
```pip install feather-format```

## How to use Feather
```
import feather
 
path = 'data.feather'
 
 # To write a feather file
feather.write_dataframe(df, path)

# To load a feather file
df = feather.read_dataframe(path)
```