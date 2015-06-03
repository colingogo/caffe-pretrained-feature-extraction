# -*- coding: utf-8 -*-

import numpy as np
import caffe
import hickle as hkl
import cv2

class FeatExtractor:
    def __init__(self, model_path, pretrained_path, blob, crop_size, meanfile_path=None, mean_values=None):
        caffe.set_mode_gpu()
        self.model_path = model_path
        self.pretrained_path = pretrained_path
        self.blob = blob
        self.crop_size = crop_size
        self.meanfile_path = meanfile_path
        self.mean_values = mean_values
        # create network
        self.net = caffe.Net(self.model_path, self.pretrained_path, caffe.TEST)
        self.net.blobs["data"].reshape(1, 3, self.crop_size, self.crop_size)
        # mean
        if self.meanfile_path is not None:
            # load mean array
            self.mean = np.load(self.meanfile_path) # expect that shape = (1, C, H, W)
            self.mean = self.mean[0]
            self.mean = self.crop_matrix(self.mean, crop_size=self.crop_size)
        elif self.mean_values is not None:
            # create mean array
            assert len(self.mean_values) == 3
            self.mean = np.zeros((3, self.crop_size, self.crop_size))
            self.mean[0] = mean_values[0]
            self.mean[1] = mean_values[1]
            self.mean[2] = mean_values[2]
        else:
            raise Exception
        # create preprocessor
        # We expect that input shape is HxWxC, and color order is RGB
        self.transformer = caffe.io.Transformer({"data": self.net.blobs["data"].data.shape})
        self.transformer.set_transpose("data", (2,0,1))
        self.transformer.set_mean("data", self.mean)
        self.transformer.set_raw_scale("data", 255)
        self.transformer.set_channel_swap("data", (2,1,0))

    def extract_feature(self, img):
        preprocessed_img = self.transformer.preprocess("data", img)
        out = self.net.forward_all(**{self.net.inputs[0]: preprocessed_img, "blobs": [self.blob]})
        feat = out[self.blob]
        feat = feat[0] 
        return feat

    def crop_matrix(self, matrix, crop_size):
        """
        :param matrix numpy.ndarray: matrix, shape = [C,H,W]
        :param crop_size integer: cropping size
        :return: cropped matrix
        :rtype: numpy.ndarray, shape = [C,H,W]
        """
        assert matrix.shape[1] == matrix.shape[2]
        corner_size = matrix.shape[1] - crop_size
        corner_size = np.floor(corner_size / 2)
        res = matrix[:, corner_size:crop_size+corner_size, corner_size:crop_size+corner_size] 
        return res
    
def create_dataset(net, datalist, dbprefix):
    with open(datalist) as fr:
        lines = fr.readlines()
    lines = [line.rstrip() for line in lines]
    feats = []
    labels = []
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
    alexnet = FeatExtractor(
                model_path="alexnet_deploy.prototxt",
                pretrained_path="alexnet.caffemodel",
                blob="fc6",
                crop_size=227,
                meanfile_path="imagenet_mean.npy"
                )
    create_dataset(net=alexnet, datalist="train.txt", dbprefix="alexnet_train")
    create_dataset(net=alexnet, datalist="test.txt", dbprefix="alexnet_test")

def run_vgg16_fc7():
    vgg16 = FeatExtractor(
            model_path="vgg16_deploy.prototxt",
            pretrained_path="vgg16.caffemodel",
            blob="fc7",
            crop_size=224,
            mean_values=[103.939, 116.779, 123.68]
            )
    create_dataset(net=vgg16, datalist="train.txt", dbprefix="vgg16_fc7_train")
    create_dataset(net=vgg16, datalist="test.txt", dbprefix="vgg16_fc_7test")

def run_vgg16_fc6():
    vgg16 = FeatExtractor(
            model_path="vgg16_deploy.prototxt",
            pretrained_path="vgg16.caffemodel",
            blob="fc6",
            crop_size=224,
            mean_values=[103.939, 116.779, 123.68]
            )
    create_dataset(net=vgg16, datalist="train.txt", dbprefix="vgg16_fc6_train")
    create_dataset(net=vgg16, datalist="test.txt", dbprefix="vgg16_fc6_test")

def run_googlenet():
    googlenet = FeatExtractor(
            model_path="googlenet_deploy.prototxt",
            pretrained_path="googlenet.caffemodel",
            blob="pool5/7x7_s1",
            crop_size=224,
            mean_values=[104.0, 117.0, 123.0]
            )
    create_dataset(net=googlenet, datalist="train.txt", dbprefix="googlenet_train")
    create_dataset(net=googlenet, datalist="test.txt", dbprefix="googlenet_test")

if __name__ == "__main__":
    run_alexnet()
    run_vgg16_fc7()
    run_vgg16_fc6()
    run_googlenet()
