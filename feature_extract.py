# -*- coding: utf-8 -*-

import numpy as np
import caffe

def crop_mean(mean, image_size, crop_size):
    corner_size = image_size - crop_size # 256 - 224 = 32
    corner_size = np.floor(corner_size / 2) # 32 / 2 = 16
    res = mean[:, corner_size:crop_size+corner_size, corner_size:crop_size+corner_size] # (:, 16:240, 16:240)
    return res

def alexnet_feature(model_path, pretrained_path, meanfile_path, images):
    caffe.set_mode_gpu()

    # laod mean array
    mean = np.load(meanfile_path) # expect that shape = (1, C, H, W)
    mean = mean[0]
    mean = crop_mean(mean, image_size=256, crop_size=227)
    
    # create network
    net = caffe.Net(model_path, pretrained_path, caffe.TEST)
    net.blobs["data"].reshape(1, 3, 227, 227)
    
    # Debug
    # for k, v in net.blobs.items():
    #     print k, ":", v.data.shape
    # for k, v in net.params.items():
    #     print k, ":", v[0].data.shape
    
    # create preprocessor
    transformer = caffe.io.Transformer({"data": net.blobs["data"].data.shape})
    transformer.set_transpose("data", (2,0,1))
    transformer.set_mean("data", mean)
    transformer.set_raw_scale("data", 255)
    transformer.set_channel_swap("data", (2,1,0))
   
    # extract features from images
    feats = []
    for img in images:
        preprocessed_img = transformer.preprocess("data", img)
        #print "*", preprocessed_img.shape
        out = net.forward_all(**{net.inputs[0]: preprocessed_img, "blobs": ["fc6"]})
        feat = out["fc6"]
        feat = feat[0] 
        feats.append(feat)
    feats = np.asarray(feats)
    return feats

if __name__ == "__main__":
    print "==> loading images and labels"
    images, labels = load_images("datalist.txt")
    print "==> extracting features"
    feats = alexnet_feature(
                model_path="alexnet_deploy.prototxt",
                pretrained_path="alexnet.caffemodel",
                meanfile_path="imagenet_mean.npy",
                images=[img1, img2]
                )
    print feats.shape
     
