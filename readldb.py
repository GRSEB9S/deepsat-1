#! /usr/bin/env python

import plyvel
import argparse
import numpy as np
from caffe.proto import caffe_pb2

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,description="Converts data and label csvs to leveldb ready for caffe")
parser.add_argument("--dbname", required=True)
args = parser.parse_args()

# Open database

db=plyvel.DB(args.dbname, create_if_missing=True)
labels=[]
for k,v in db.iterator():
    datum = caffe_pb2.Datum()
    
    datum.ParseFromString(v)
    labels.append(datum.label)
    
print "Unique labels in database : %s = %s" % (args.dbname, np.unique(labels))