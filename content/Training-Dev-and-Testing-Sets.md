## Train set, Cross Validation set (Dev set) and Test set
You want to have 3 different sets:

1. Train set: data used to train your network
2. Cross Validation set: used to test algorithms while deciding on the architecture of the network
3. Test set: this is the secret set, and the network should never see this data during training, you can only use this set once the network was trained to see how it performs with new (never seen) data

For small datasets (100 to 10000) the ratio is normally 
1. 60% for the training set
2. 20% for the cross validation set
3. 20% for the test set

For bigger datasets 
1. 98% for the training set
2. 1% for the cross validation set
3. 1% for the test set

