# MiniBatchSVM
Mini-batch SVM / Logistic Regresion, Online SVM training for large scale data 

## Usage

```
./MiniBatchSVM.py 
```
By default: run mnist with SVM, batchsize=1000, epoch=20.
```
./MiniBatchSVM.py --model log
```
By default: run mnist with Logistic Regresion, batchsize=1000, epoch=20.

You can train your own data by specific the options

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

## Example

    $ ./MiniBatchSVM.py  
```
Using Hinge Loss SVM...
Epoch 1/20
   Test Score 0.8463 BestScore 0.8463
Epoch 2/20
   Test Score 0.8507 BestScore 0.8507
Epoch 3/20
   Test Score 0.8707 BestScore 0.8707
Epoch 4/20
   Test Score 0.8441 BestScore 0.8707
Epoch 5/20
   Test Score 0.8636 BestScore 0.8707
Epoch 6/20
   Test Score 0.8493 BestScore 0.8707
Epoch 7/20
   Test Score 0.8541 BestScore 0.8707
Epoch 8/20
   Test Score 0.8134 BestScore 0.8707
Epoch 9/20
   Test Score 0.8756 BestScore 0.8756
Epoch 10/20
   Test Score 0.8282 BestScore 0.8756
Epoch 11/20
   Test Score 0.8607 BestScore 0.8756
Epoch 12/20
   Test Score 0.8482 BestScore 0.8756
Epoch 13/20
   Test Score 0.8442 BestScore 0.8756
Epoch 14/20
   Test Score 0.8599 BestScore 0.8756
Epoch 15/20
   Test Score 0.8737 BestScore 0.8756
Epoch 16/20
   Test Score 0.8228 BestScore 0.8756
Epoch 17/20
   Test Score 0.8561 BestScore 0.8756
Epoch 18/20
   Test Score 0.8688 BestScore 0.8756
Epoch 19/20
   Test Score 0.8635 BestScore 0.8756
Finished, The finally best score is: 0.8756
```


## Logistic Regresion

## SGD

## Mini-batch


