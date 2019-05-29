# CSP PyTorch Implementation
Unofficially Pytorch implementation of [**High-level Semantic Feature Detection: A New Perspective for Pedestrian Detection**](<https://github.com/liuwei16/CSP>)

This code is only for CityPersons dataset, and only for center-position+height regression+offset regression model.

## Requirement

Python, pytorch and other related libaries

GPU is needed

## Usage

Compile lib

~~~
cd util
make all
~~~

Prepare CityPersons dataset as the original codes doing

* For citypersons, we use the training set (2975 images) for training and test on the validation set (500 images), we assume that images and annotations are stored in  `./data/citypersons`, and the directory structure is

```
*DATA_PATH
	*annotations
		*anno_train.mat
		*anno_val.mat
	*images
		*train
		*val
```



Training & val

~~~
python trainval_torchstyle.py
python trainval_caffestyle.py
~~~

NOTE

using caffe style, you need to download additional pre-trained weight.

## update

12.56 MR trained weight provided here

https://www.dropbox.com/s/2uivsotq46la15u/CSPNet-89.pth?dl=0