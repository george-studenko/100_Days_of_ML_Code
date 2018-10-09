I've been working on the facial keypoints detection project, read this paper [Facial Key Points Detection using Deep Convolutional Neural Network - NaimishNet](https://arxiv.org/pdf/1710.00977.pdf) and took it as a starting point for my network architecture.

I'm using MSELoss as my loss function and Adam with default parameters as the optimizer (LR=0.001, \beta=(0.9,0.999)) and now trying to get the best performance possible on the network.

This is my architecture so far: [models.py](code/cv/models.py)

This is the best I could get so far:

```# With MSELoss and optim.SGD(net.parameters(),0.001,momentum=0.9,nesterov=True)
Epoch: 1, Batch: 10, Avg. Loss: 0.028066457714885473
Epoch: 1, Batch: 20, Avg. Loss: 0.02104236278682947
Epoch: 1, Batch: 30, Avg. Loss: 0.022216709144413472
Epoch: 1, Batch: 40, Avg. Loss: 0.029279047250747682
Epoch: 1, Batch: 50, Avg. Loss: 0.01847806265577674
Epoch: 1, Batch: 60, Avg. Loss: 0.024008906725794078
Epoch: 1, Batch: 70, Avg. Loss: 0.022673950437456368
Epoch: 1, Batch: 80, Avg. Loss: 0.025031224079430103
Epoch: 1, Batch: 90, Avg. Loss: 0.035366627294570206
Epoch: 1, Batch: 100, Avg. Loss: 0.016411050129681824
Epoch: 1, Batch: 110, Avg. Loss: 0.0172842089086771
Epoch: 1, Batch: 120, Avg. Loss: 0.02427839869633317
Epoch: 1, Batch: 130, Avg. Loss: 0.023034661076962948
Epoch: 1, Batch: 140, Avg. Loss: 0.022957767266780137
Epoch: 1, Batch: 150, Avg. Loss: 0.016886541806161403
Epoch: 1, Batch: 160, Avg. Loss: 0.02071635639294982
Epoch: 1, Batch: 170, Avg. Loss: 0.018030618131160737
Epoch: 1, Batch: 180, Avg. Loss: 0.022987023554742338
Epoch: 1, Batch: 190, Avg. Loss: 0.02365094544366002
Epoch: 1, Batch: 200, Avg. Loss: 0.021768472623080015
Epoch: 1, Batch: 210, Avg. Loss: 0.022931396309286357
Epoch: 1, Batch: 220, Avg. Loss: 0.020561089925467967
Epoch: 1, Batch: 230, Avg. Loss: 0.02189907068386674
Epoch: 1, Batch: 240, Avg. Loss: 0.023978890758007763
Epoch: 1, Batch: 250, Avg. Loss: 0.01856830148026347
Epoch: 1, Batch: 260, Avg. Loss: 0.019809638615697622
Epoch: 1, Batch: 270, Avg. Loss: 0.02397430771961808
Epoch: 1, Batch: 280, Avg. Loss: 0.027975480444729327
Epoch: 1, Batch: 290, Avg. Loss: 0.02225145073607564
Epoch: 1, Batch: 300, Avg. Loss: 0.021185639034956693
Epoch: 1, Batch: 310, Avg. Loss: 0.052688654605299236
Epoch: 1, Batch: 320, Avg. Loss: 0.02006478551775217
Epoch: 1, Batch: 330, Avg. Loss: 0.01888055559247732
Epoch: 1, Batch: 340, Avg. Loss: 0.027196702361106873
Epoch: 2, Batch: 10, Avg. Loss: 0.017323469463735818
Epoch: 2, Batch: 20, Avg. Loss: 0.01812070393934846```


