# MiniBatchSVM
Mini-batch SVM / Logistic Regresion, Online SVM training for large scale data 

## Usage

```
./MiniBatchSVM.py 
```
By default: run mnist with batchsize=1000, epoch=20
You can use it to train your own data by specific the options

```
./MiniBatchSVM.py [options] [--trianlist path-to-training-data-list-file] 
```

## Options
```
  -h, --help            show this help message and exit
  -m MODEL, --model=MODEL
                        svm, log
  -t TRAINLIST, --trainlist=TRAINLIST
                        trainlist file
  -T TESTLIST, --testlist=TESTLIST
                        testlist file
  -b BATCHSIZE, --batchsize=BATCHSIZE
                        batch size
  -e EPOCH, --epoch=EPOCH
                        max epoch
  -c NCLASSES, --nclasses=NCLASSES
                        num of the class
  -n, --norm            do mean normalization
  -l, --labelstart1     use this option when  the label of your data is bengin
                        at 1
                        
```

## SVM


## Logistic Regresion

## SGD

## Mini-batch


