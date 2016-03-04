#! /usr/bin/env python

import argparse
import numpy as np
import matplotlib.pyplot as plt
import string

def read_csv(csv):
    """ Reads a csv (output of caffe/extra/tools/parse_logs.py) and returns dict {column name : numpy array of data} """
    f = open(csv,'r')
    lines = f.readlines()
    cols = map(string.strip, lines.pop(0).split(" "))

    column_data = dict.fromkeys(cols, [])
    print column_data
    for l in lines:
        i=0
        print l.split("\t")
        for val in l.split("\t"):
            column_data[cols[i]].append(val) #append to list in the respective column
            i += 1
    
    for key in column_data.keys():
        column_data[key] = np.array(column_data[key])
        
    return column_data
    

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,description="Converts data and label csvs to leveldb ready for caffe")
parser.add_argument("-i",dest="csv", default="log.train")
parser.add_argument("--cols", default=["#Iters", "TrainingLoss"])
args = parser.parse_args()

column_data = read_csv(args.csv)
plt.plot(column_data[args.cols[0]], column_data[args.cols[1]])
plt.show()
