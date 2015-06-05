# caffe-pretrained-feature-extraction

- Go to directory
```
cd /path/to/caffe-pretrained-feature-extraction
```

- Download the pre-trained weights of AlexNet
```
$CAFFE_ROOT/scripts/download_model_binary.py $CAFFE_ROOT/models/bvlc_alexnet
cp $CAFFE_ROOT/models/bvlc_alexnet/bvlc_alexnet.caffemodel ./alexnet.caffemodel
$CAFFE_ROOT/data/ilsvrc12/get_ilsvrc_aux.sh
cp $CAFFE_ROOT/data/ilsvrc12/imagenet_mean.binaryproto .
python convert_to_imagenet_mean_npy.py
```

- Download the pre-trained weights of VGG(16 layers)
```
$CAFFE_ROOT/scripts/download_model_from_gist.sh 211839e770f7b538e2d8 $CAFFE_ROOT/models
cd $CAFFE_ROOT/models
mv 211839e770f7b538e2d8 VGG_ILSVRC_16_layers
cd VGG_ILSVRC_16_layers
wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
cp VGG_ILSVRC_16_layers.caffemodel /path/to/caffe-pretrained-feature-extraction/vgg16.caffemodel
cd /path/to/caffe-pretrained-feature-extraction
```

- Download the pre-trained weights of GoogLeNet
```
$CAFFE_ROOT/scripts/download_model_binary.py  $CAFFE_ROOT/models/bvlc_googlenet
cp $CAFFE_ROOT/models/bvlc_googlenet/bvlc_googlenet.caffemodel ./googlenet.caffemodel
```

- Create train.txt, test.txt. The contents are like this:
```
cat train.txt
/path/to/trainimage0 1
/path/to/trainimage1 0
(...)
cat test.txt
/path/to/testimage0 0
/path/to/testimage1 0
(...)
```

- Extract feature vectors from AlexNet(fc6), VGG(fc7, fc6), GoogLeNet(pool5/7x7_s1) and dump to hickle files
```
python example.py
python google_process_features.py
```

- Train linear SVM, and report prediction on test data
```
python train_test_predict.py
cat alexnet_predict.txt
id,label
123,0
456,1
789,0
(...)
```
