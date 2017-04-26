# MiniBatchSVM
Mini-batch SVM / Logistic Regresion, Online SVM training for large scale data 

## Usage

```
./MiniBatchSVM.py [options] # default: run mnist with batchsize=1000, epoch=20
./MiniBatchSVM.py [options] [--trianlist path-to-training-data-list-file] 
```

## Options
```
  -h, --help            show this help message and exit
  -m MODEL, --model=MODEL
                        svm, log
  -l, --labelstart1     use this option when  the label of your data is bengin
                        at 1
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

```

## SVM


## Logistic Regresion

## SGD

## Mini-batch


