# mnist-h5

```python
>>> import h5py
>>> D=h5py.File("data_0.h5","r")
>>> D['data'].shape
(60000, 1, 28, 28)
>>> D['label'].shape
(60000,)
>>>                 
```
