#! /usr/bin/env python
from pyparsing import _PositionToken

import caffe
import plyvel
import argparse
import numpy as np
from caffe.proto import caffe_pb2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

__author__ = "Mohamed Ezz"


parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,description="Converts data and label csvs to leveldb ready for caffe")
parser.add_argument("--data", required=True)
parser.add_argument("--dbname")
args = parser.parse_args()

# Read csv
data=np.genfromtxt(args.data, dtype='uint8', delimiter=',', skip_header=True, missing_values="5")

labels = data[:,-1]
data = data[:,:-1]
print "Labels : ", np.unique(labels)

if args.dbname:
    # Open database
    db=plyvel.DB(args.dbname, create_if_missing=True)
    dbbatch = db.write_batch()
    for i in range(data.shape[0]):
        datum = caffe_pb2.Datum()
        
        datum.channels = 1
        datum.height=1
        datum.width= data.shape[1]
        datum.label = int(labels[i])
        datum.data = data[i,:].tostring()
        
        dbbatch.put('{:05}'.format(i),datum.SerializeToString())
    
    dbbatch.write()

#======================================
print 'Baseline with random forest and Logistic Regression'

N = data.shape[0]
Ntrain = int(N*0.7)
traindata = data[:Ntrain,:] 

trainlabels = labels[:Ntrain]
testdata = data[Ntrain:,:]
testlabels = labels[Ntrain:]

############## RANDOM FOREST
clf = RandomForestClassifier(n_estimators=20, min_samples_split=1, random_state=123, class_weight="balanced", n_jobs=3)
clf.fit(traindata,trainlabels)
testpred = clf.predict(testdata)
trainpred= clf.predict(traindata)
print "RandomForest Accuracy :  Train : %.3f , Test: %.3f" % (accuracy_score(trainlabels, trainpred), accuracy_score(testlabels, testpred))

############## LOGISTIC REGRESSION
clf = LogisticRegression( C=1000000, intercept_scaling=1, class_weight="balanced", random_state=123)
clf.fit(traindata,trainlabels)
testpred = clf.predict(testdata)
trainpred= clf.predict(traindata)
print "Logistic Accuracy :  Train : %.3f , Test: %.3f" % (accuracy_score(trainlabels, trainpred), accuracy_score(testlabels, testpred))

