# -*- coding: utf-8 -*-

import numpy as np
import plyvel
import caffe
from caffe.proto import caffe_pb2

def convert_images_to_leveldb(datalist, dbname):
    db = plyvel.DB(dbname, create_if_missing=True)
    datum = caffe_pb2.Datum()
    wb = db.write_batch()

    with open(datalist) as fr:
        lines = fr.readlines()
    lines = [line.rstrip() for line in lines]
    
    item_id = 0
    for line_i, line in enumerate(lines):
        path, label = line.split()
        label = int(label)

        img = caffe.io.load_image(path)
        datum = caffe.io.array_to_datum(img, label)
        wb.put("{:0>8d}".format(item_id), datum.SerializeToString())
        item_id += 1
        if (item_id + 1) % 1000 == 0:
            wb.write()
            wb = db.write_batch()
            print item_id + 1
    if (item_id + 1) % 1000 != 0:
        wb.write()
        print "last batch"
    db.close()

if __name__ == "__main__":
    convert_images_to_leveldb("datalist.txt", "dogs-vs-cats-leveldb")

