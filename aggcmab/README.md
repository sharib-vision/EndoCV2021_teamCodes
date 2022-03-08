# endocv21
This repository has the code corresponding to our EndoCV 2021 submission, which is associated to the paper:

```
Multi-Center Polyp Segmentation withDouble Encoder-Decoder Networks
Adrian Galdran, Gustavo Carneiro, Miguel A. González Ballester
EndoCV 2021 Workshop on Computer Vision in Endoscopy 2021
```

that you can find [here](http://ceur-ws.org/Vol-2886/paper1.pdf). 

The interesting bit here is that we force a neural network to sample data from a multi-site dataset in such a way that in a batch all centers are as represented as possible. If you are interested in doing something similar on your own work, please check out the `prepare_endo_data.py` file, where we preprocess the data so that we oversample minority centers in our training set, and then the dataloader in `utils/get_loaders.py`, which provides batches with centers on it, allowing us to monitor per-center performance during training (see `train_centers.py` for this).

