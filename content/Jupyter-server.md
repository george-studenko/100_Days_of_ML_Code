## Configur a Jupyter Notebook Server
### Configure the server
1. On the server you first create the config with:  
```jupyter notebook --generate-config```  

1. Then open the file that was created and change these 2 settings:  

```c.NotebookApp.allow_origin = '*' #allow all origins```    
  
```c.NotebookApp.ip = '0.0.0.0' # listen on all IPs```  

1. Save the file.  

1. Set a password to protect the server with:  
```jupyter notebook password```   

1. Start the server with:  
```jupyter notebook --ip 0.0.0.0 --port 8888```

1. Open a browser on the client machine and open the url:  

```192.168.1.100:8888```

You might need to open the port 8888 on the server if the firewall is blocking it, for ubuntu run:  
```sudo ufw allow 8888```  

This will enable tcp:8888 port, which is ur default jupyter port. 

To access it remotely you will need to forward the port on the router to the server IP. I recommend configuring an static IP on the server or set a DHCP rule in the router so it assigns the same IP to the server every time.