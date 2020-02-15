# CIDH-caffe
We fork the repository from [Caffe](https://github.com/BVLC/caffe) and make our modifications. 
## Environment
* caffe
* python 2.7

## Data Preparation

You can download the data set and labels from [UCMerced-4](https://mp.weixin.qq.com/s/6I-eBg2_m-T0ugBBHG1yEg). The password is `ti2v`. The train label file and test label file is `train_label.txt` and `test_label.txt`, respectively.

After download the data set, you will change the data set path in label files. 

For example, the path in my train label file is `/home/lrh/dataset/UCdataset-4/agricultural00.jpg 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0` where `1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0` is `agricultural00.jpg`'s label, and you need to replace the path of the data set with your path `your path/UCdataset-4/agricultural00.jpg 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0`.
   
## Test the model
You can download the pretrained model [here](https://pan.baidu.com/s/1iO72cD953TknxeJLhuhrtA). The password is `d4uo`. You need to put the trained model `ResUCMD32.caffemodel` in `./models/Resnet-50/`. 

In `./models/ResNet-50/predict/`, we give a test python file `predict_parallel.py` to show how to evaluate the trained hash model. 

`python predict_parallel.py`
