# TFprocessing

A demonstration of using Tensor Flow for high performance distributed computing of a computational graph, with nodes that are custom ops for beamforming and 4D blood velocity estimation.

The version of Tensorflow used was 1.1 deployed on python 2.7.12 with the following modules:
- `h5py/2.7.1-python-2.7.12_ucs4`
- `numpy/1.13.3-python-2.7.12_ucs4-openblas-0.2.20`
- `scipy/0.19.1-numpy-1.13.3-python-2.7.12_ucs4`

Unfortunately, version 2 of Tensorflow has made major updates to the queues mechanism that this code relies on. These might imply a major effort to migrate to. Nevertheless, it shouldn't have any major problem running in any 1.x version. 

## Python 2.7

```bash
conda env create -f environment.yml
conda activate tf11
```

## Test Tensorflow

```python
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
```

## Compile C source code

(your include path below will vary!)

```bash
g++ -v -std=c++11 -mavx -fopenmp -O2 -shared ultrasound.cc -o ultrasound.so -I /scratch/dpb6/anaconda3/envs/tf11/lib/python2.7/site-packages/tensorflow/include -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -lfftw3 -lm
```

## Run bf_local

```bash
python bf_local.py --worker_hosts=localhost:2500 --job_name=worker --task_index=0
```

