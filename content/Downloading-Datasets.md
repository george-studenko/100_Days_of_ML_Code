## Kaggle
To Download datasets from Kaggle you can use the:  
```kaggle-cli``` 

You can install it with pip:  
```pip install kaggle-cli```  

or to upgrade it using:  
```pip install kaggle-cli --upgrade```

The only problem with it is that it will download all of the dataset files, which can be huge (more than 20 GBs).

Another way to do it is with the [CurlWGet](https://chrome.google.com/webstore/detail/curlwget/jmocjfidanebdlinpbcdkcmgdifblncg?hl=en) Chrome extension, that way you can download only the file you want directly into your project folder.

You can also use symbolic links to set your data folder in different drives or outside of your project folder.

```ls -l``` will show where folders and links are pointing to.

### How to submit it to kaggle

Kaggle will provide with the strucutre of the csv to create and submit:

You can create a csv file using Pandas   
```df.to_csv('file.gz', compression='gzip', index=False)```  




