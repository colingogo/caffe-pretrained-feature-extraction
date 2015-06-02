# -*- coding: utf-8 -*-

import numpy as np
import caffe
import hickle as hkl
import cv2

class AlexNet:
    def __init__(self, model_path, pretrained_path, meanfile_path):
        caffe.set_mode_gpu()
        # load mean array
        self.mean = np.load(meanfile_path) # expect that shape = (1, C, H, W)
        self.mean = self.mean[0]
        self.mean = self.crop_matrix(self.mean, matrix_size=256, crop_size=227)
    
        # create network
        self.net = caffe.Net(model_path, pretrained_path, caffe.TEST)
        self.net.blobs["data"].reshape(1, 3, 227, 227)
    
        # create preprocessor
        self.transformer = caffe.io.Transformer({"data": self.net.blobs["data"].data.shape})
        self.transformer.set_transpose("data", (2,0,1))
        self.transformer.set_mean("data", self.mean)
        self.transformer.set_raw_scale("data", 255)
        self.transformer.set_channel_swap("data", (2,1,0))

    def extract_feature(self, img, blob="fc6"):
        # expect img.shape = HxWxC and colors = RGB
        preprocessed_img = self.transformer.preprocess("data", img)
        out = self.net.forward_all(**{self.net.inputs[0]: preprocessed_img, "blobs": [blob]})
        feat = out[blob]
        feat = feat[0] 
        return feat

    def crop_matrix(self, matrix, matrix_size, crop_size):
        corner_size = matrix_size - crop_size # 256 - 224 = 32
        corner_size = np.floor(corner_size / 2) # 32 / 2 = 16
        res = matrix[:, corner_size:crop_size+corner_size, corner_size:crop_size+corner_size] # (:, 16:240, 16:240)
        return res

class VGG16:
    def __init__(self, model_path, pretrained_path, mean=None):
        caffe.set_mode_gpu()
        
        # create network
        self.net = caffe.Net(model_path, pretrained_path, caffe.TEST)
        self.net.blobs["data"].reshape(1, 3, 224, 224)

        # hyper params for preprocessor
        if mean is None:
            self.mean = [103,939, 116.779, 123.68]
        else:
            self.mean = mean
        self.img_size = 256
        self.crop_size = 224
 
    def preprocess(self, img):
        """
        expect img.shape = HxWxC and colors = RGB
        """  
        # resize
        # if img.shape[0] < img.shape[1]:
        #     dsize = (int(np.floor(float(self.img_size)*img.shape[1]/img.shape[0])), self.img_size)
        # else:
        #     dsize = (img_size, int(np.floor(float(img_size)*img.shape[0]/img.shape[1])))
        # img = cv2.resize(img, dsize, interpolation=cv2.INTER_CUBIC)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
        assert img.shape == (self.img_size, self.img_size, 3)
        # convert to float32 
        img = img.astype("float32", copy=False)
        # crop
        corner_size = self.img_size - self.crop_size
        corner_size = np.floor(corner_size / 2)
        img = img[corner_size:self.crop_size+corner_size, corner_size:self.crop_size+corner_size] 
        assert img.shape == (self.crop_size, self.crop_size, 3)
        # subtract mean
        for c in xrange(3):
            img[:,:,c] = img[:,:,c] - self.mean[c]
        # reorder axis
        img = np.rollaxis(img, 2, 0)
        assert img.shape == (3, self.crop_size, self.crop_size)
        return img

    def extract_feature(self, img, blob="fc6"):
        """
        expect img.shape = HxWxC and colors = RGB
        """
        preprocessed_img = self.preprocess(img)
        out = self.net.forward_all(**{self.net.inputs[0]: preprocessed_img, "blobs": [blob]})
        feat = out[blob]
        feat = feat[0] 
        return feat

def create_dataset(net, datalist, dbprefix):
    with open(datalist) as fr:
        lines = fr.readlines()
    lines = [line.rstrip() for line in lines]
    
    feats = []
    labels = []
    batch_id = 0
    for line_i, line in enumerate(lines):
        img_path, label = line.split()

        img = caffe.io.load_image(img_path)
        feat = net.extract_feature(img)
        feats.append(feat)

        label = int(label)
        labels.append(label)

        if (line_i + 1) % 100 == 0:
            print "processed", line_i + 1

    feats = np.asarray(feats)
    labels = np.asarray(labels)
    hkl.dump(feats, dbprefix + "_features.hkl", mode="w")
    hkl.dump(labels, dbprefix + "_labels.hkl", mode="w")

def run_alexnet():
    alexnet = AlexNet(
                model_path="alexnet_deploy.prototxt",
                pretrained_path="alexnet.caffemodel",
                meanfile_path="imagenet_mean.npy"
                )
    create_dataset(net=alexnet, datalist="train.txt", dbprefix="alexnet_train")
    create_dataset(net=alexnet, datalist="test.txt", dbprefix="alexnet_test")

def run_vgg16():
    vgg16 = VGG16(
            model_path="vgg16_deploy.prototxt",
            pretrained_path="vgg16.caffemodel",
            )
    create_dataset(net=vgg16, datalist="train.txt", dbprefix="vgg16_train")
    create_dataset(net=vgg16, datalist="test.txt", dbprefix="vgg16_test")

def run_googlenet():
    googlenet = GoogLeNet(
            model_path="googlenet_deploy.prototxt",
            pretrained_path="googlenet.caffemodel",
            meanfile_path="imagenet_mean.npy"
            )
    create_dataset(net=googlenet, datalist="train.txt", dbprefix="googlenet_train")
    create_dataset(net=googlenet, datalist="test.txt", dbprefix="googlenet_test")

if __name__ == "__main__":
    #run_alexnet()
    run_vgg16()
    #run_googlenet()
