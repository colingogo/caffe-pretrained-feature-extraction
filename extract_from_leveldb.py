# -*- coding: utf-8 -*-

import plyvel
import caffe
from caffe.proto import caffe_pb2

if __name__ == "__main__":
    db = plyvel.DB("alexnet-features")
    datum = caffe_pb2.Datum.FromString(db.Get("1"))
    arr = caffe.io.datum_to_array(datum)
    print arr
    db.close()
