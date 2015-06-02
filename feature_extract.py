# -*- coding: utf-8 -*-

import numpy as np
import caffe
import hickle as hkl
import cv2

###################################
class AlexNet:
    def __init__(self, model_path, pretrained_path, meanfile_path):
        """
        :param model_path string: path to model defined file (such as deploy.prototxt)
        :param pretrained_path string: path to pretrained data (such as *.caffemodel)
        :param meanfile_path string: path to mean data (such as imagenet_mean.npy)
        """
        caffe.set_mode_gpu()

        # load mean array
        self.mean = np.load(meanfile_path) # expect that shape = (1, C, H, W)
        self.mean = self.mean[0]
        self.mean = self.crop_matrix(self.mean, matrix_size=256, crop_size=227)
    
        # create network
        self.net = caffe.Net(model_path, pretrained_path, caffe.TEST)
        self.net.blobs["data"].reshape(1, 3, 227, 227)
    
        # create preprocessor (expect input: HxWxC(RGB))
        self.transformer = caffe.io.Transformer({"data": self.net.blobs["data"].data.shape})
        self.transformer.set_transpose("data", (2,0,1))
        self.transformer.set_mean("data", self.mean)
        self.transformer.set_raw_scale("data", 255)
        self.transformer.set_channel_swap("data", (2,1,0))

    def extract_feature(self, img, blob="fc6"):
        """
        :param img numpy.ndarray: image data to extract feature, shape = [H,W,C] order RGB
        :param blob string: blob name to extract feature
        :return: d-dimensional feature vector
        :rtype: numpy.ndarray
        """
        preprocessed_img = self.transformer.preprocess("data", img)
        out = self.net.forward_all(**{self.net.inputs[0]: preprocessed_img, "blobs": [blob]})
        feat = out[blob]
        feat = feat[0] 
        return feat

    def crop_matrix(self, matrix, matrix_size, crop_size):
        """
        :param matrix numpy.ndarray: matrix, shape = [C,H,W]
        :param matrix_size integer: size of matrix
        :param crop_size integer: cropping size
        :return: cropped matrix
        :rtype: numpy.ndarray, shape = [C,H,W]
        """
        corner_size = matrix_size - crop_size
        corner_size = np.floor(corner_size / 2)
        res = matrix[:, corner_size:crop_size+corner_size, corner_size:crop_size+corner_size] 
        return res

###################################
class VGG16:
    def __init__(self, model_path, pretrained_path):
        caffe.set_mode_gpu()
        
        # create mean array
        self.mean = np.zeros((3, 224, 224))
        self.mean[0] = 103.939
        self.mean[1] = 116.779
        self.mean[2] = 123.68
        
        # create network
        self.net = caffe.Net(model_path, pretrained_path, caffe.TEST)
        self.net.blobs["data"].reshape(1, 3, 224, 224)

        # create preprocessor (expect input: HxWxC(RGB))
        self.transformer = caffe.io.Transformer({"data": self.net.blobs["data"].data.shape})
        self.transformer.set_transpose("data", (2,0,1))
        self.transformer.set_mean("data", self.mean)
        self.transformer.set_raw_scale("data", 255)
        self.transformer.set_channel_swap("data", (2,1,0))

    def extract_feature(self, img, blob="fc7"):
        preprocessed_img = self.transformer.preprocess("data", img)
        out = self.net.forward_all(**{self.net.inputs[0]: preprocessed_img, "blobs": [blob]})
        feat = out[blob]
        feat = feat[0] 
        return feat

###################################
class GoogLeNet:
    def __init__(self, model_path, pretrained_path):
        caffe.set_mode_gpu()

        # create mean array
        self.mean = np.zeros((3, 224, 224))
        self.mean[0] = 104.0
        self.mean[1] = 117.0
        self.mean[2] = 123.0

        # create network
        self.net = caffe.Net(model_path, pretrained_path, caffe.TEST)
        self.net.blobs["data"].reshape(1, 3, 224, 224)
    
        # create preprocessor
        self.transformer = caffe.io.Transformer({"data": self.net.blobs["data"].data.shape})
        self.transformer.set_transpose("data", (2,0,1))
        self.transformer.set_mean("data", self.mean)
        self.transformer.set_raw_scale("data", 255)
        self.transformer.set_channel_swap("data", (2,1,0))

    def extract_feature(self, img, blob="pool5/7x7_s1"):
        # expect img.shape = HxWxC and colors = RGB
        preprocessed_img = self.transformer.preprocess("data", img)
        out = self.net.forward_all(**{self.net.inputs[0]: preprocessed_img, "blobs": [blob]})
        feat = out[blob]
        feat = feat[0] 
        return feat
    
###################################
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

###################################
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
            )
    create_dataset(net=googlenet, datalist="train.txt", dbprefix="googlenet_train")
    create_dataset(net=googlenet, datalist="test.txt", dbprefix="googlenet_test")

if __name__ == "__main__":
    #run_alexnet()
    #run_vgg16()
    run_googlenet()
