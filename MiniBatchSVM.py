#!/usr/bin/env python
"""
# Author: XU Kui xukui.cs@gmail.com    
# Created Time : Wed 19 Apr 2017 09:46:16 PM CST 
 
# File Name: MiniBatchSVM.py 
# Description:Mini-batch SVM / Logistic Regresion, Online SVM training for large scale data 
 
"""
 
 
import numpy as np
import os, sys
import h5py
from optparse import OptionParser 
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import HashingVectorizer

global model
model = ""

def meanNorm(X):
    length = len(X)
    ind=[]
    for i in range(length):
        x=X[i,0,]
        if np.max(x)-np.min(x)==0:
            ind.append(i)
        else:
            X[i,0,]=(x-np.mean(x))/(np.max(x)-np.min(x))
    return X, ind

def loadH5file(filepath, opt):
        train=h5py.File(filepath, 'r')
        if opt.labelstart1 :
            y_train=np.asarray(train['label']) - 1
        else:
            y_train=np.asarray(train['label'])

        X_train=np.asarray(train['data'])
        if opt.norm:
            X_train, ind =meanNorm(X_train)
            #print X_train.shape, len(ind)
            # remove fully zore samples 
            for i in range(len(ind)-1 ,-1,-1):
                X_train = np.delete(X_train,ind[i],0)

        #print X_train.shape
        y_train=y_train.reshape(y_train.shape[0])
        vectorNum = 1
        for i in range(1,len(X_train.shape)):
            vectorNum *= X_train.shape[i]
        X_train=X_train.reshape(X_train.shape[0], vectorNum)
        return X_train, y_train


    
#def train(model, X_train, y_train, nClasses=10, batchSize=256):
def train( X_train, y_train, nClasses=10, batchSize=256):
    best_score=0
    X_count = X_train.shape[0]
    batchCount= X_count / batchSize
    j=0
    shuffledRange = range(X_count)
    shuffledX = X_train[shuffledRange,]
    shuffledY = [y_train[i] for i in shuffledRange]

    global model
    for i in range(0, batchCount):  # Iterate over "mini-batches" of 1000 samples each
        j+=1
        y_train_batch = shuffledY[i*batchSize :(i +1)* batchSize]
        X_train_batch = shuffledX[i*batchSize :(i +1)* batchSize,]
        #vectorizer.fit_transform(train_data[i:i + batchSize])
        # Update the classifier with documents in the current mini-batch
        model.partial_fit(X_train_batch, y_train_batch, classes=range(nClasses))

#def test(model, X_test, y_test):
def test(X_test, y_test):
    global model
    score = model.score(X_test, y_test)
    return score

def createModel(modelname ="svm"):
    from sklearn.linear_model import SGDClassifier
    global model

    # SVM classifier trained online with stochastic gradient descent
    model = SGDClassifier(loss="hinge", penalty="l2")
    if modelname=="log":
        # Logistic Regresion classifier trained online with stochastic gradient descent
        model = SGDClassifier(loss="log", penalty="l2")  
        print "Using Logistic Regression..."
    else:
        print "Using Hinge Loss SVM..."

if __name__ == "__main__":
    usage = "usage: %prog [options] [--trianlist path-to-training-data-list-file] \n\
            test on mnist dataset, just type: ./MiniBatchSVM.py "

    optParser = OptionParser(usage=usage)
    optParser.add_option("-m", "--model", 
            action = "store", type = 'string', dest = "model", default = "svm", 
            help = "svm, log")
    optParser.add_option("-t", "--trainlist", action = "store", type = 'string', \
            dest = "trainlist", default = "data/mnist-h5/train.list", help = "trainlist file")
    optParser.add_option("-T", "--testlist", action = "store", type = 'string', \
            dest = "testlist", default = "data/mnist-h5/test.list", help = "testlist file")
    optParser.add_option("-b", "--batchsize", action = "store", type = 'int', \
            dest = "batchsize", default = 1000, help = "batch size")
    optParser.add_option("-e", "--epoch", action = "store", type = 'int', \
            dest = "epoch", default = 20, help = "max epoch")
    optParser.add_option("-c", "--nclasses", action = "store", type = 'int', \
            dest = "nclasses", default = 10, help = "num of the class")
    optParser.add_option("-n", "--norm", 
            action = "store_true", dest = "norm", default = False, 
            help = "do mean normalization")
    optParser.add_option("-l", "--labelstart1", 
            action = "store_false", dest = "labelstart1", default = False, 
            help = "use this option when  the label of your data is bengin at 1 ")

    (opt, args) = optParser.parse_args()

    trainFile = open(opt.trainlist,"r")
    trainFileLists = trainFile.readlines()
    testFile = open(opt.testlist,"r")
    testFileLists = testFile.readlines()
    best_score = 0 
    createModel(opt.model)
    for k in range(1,opt.epoch):
        print("Epoch %s/%s" % (k, opt.epoch))
          
        for filename in trainFileLists:
            filename=filename.strip()
            (X,Y) = loadH5file(filename, opt)
            train(X, Y, opt.nclasses, opt.batchsize)
        for filename in testFileLists:
            filename=filename.strip()
            (X,Y) = loadH5file(filename, opt)
            #score = test(model, X, Y)
            score = test(X, Y)
            if score > best_score:
                best_score = score
        #print(("   Batch %s/%s  Score %.3f BestScore %.3f")  %( j, batchCount, score, best_score))
        print(("   Test Score %.4f BestScore %.4f")  %( score, best_score))
    print(("Finished, The finally best score is: %.4f") %(best_score) )


