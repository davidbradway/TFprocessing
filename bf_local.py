# -*- coding: utf-8 -*-
"""
    To run use: 
    python bf_local.py --worker_hosts=localhost:2500 --job_name=worker --task_index=0
    
    ----------------------------------------------------------------------------
    The MIT License (MIT)
    
    Copyright (c) 2017 Alexandra Instituttet
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""
import numpy as np
import tensorflow as tf
import threading
import time
import sys
import scipy.io as sio
from core.queues import *
from core.graph_parser import *
from input_readers.input_reader_multi import *
import scipy.io
import h5py

from core.ultrasound_tensor_ops import UltrasoundOp

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "should be 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS

worker_hosts = FLAGS.worker_hosts.split(",")

# Create a cluster from the parameter server and worker hosts.
cluster = tf.train.ClusterSpec({"worker": worker_hosts})

# Create and start a server for the local task.
server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

session = tf.Session(server.target)

# ########################################DATA TO BE READ IN #########################################
# import os
# cur_dir = os.getcwd()

# Read and exec params
with open('../data/3D/params.txt', 'r') as params_file:
    params = params_file.read()

exec params

# ################################### END READ DATA ################################################

# Array elements position [x y z] - matrix array 32x32 Vermon
data = scipy.io.loadmat('../data/3D/elem_pos.mat')
element_pos = np.array(data['elem_pos'])

# Selection of points to beamform
data = scipy.io.loadmat('../data/3D/bf_points.mat')
bf_points = np.array(data['bf_points'])

# np.set_printoptions(threshold=np.nan)

# Create graph from xml file
tparser = TParser('../data/3D/bf_flow_local.xml')
tparser.parse_xml()
tparser.print_commands()

exec tparser.return_code()

# with open('data_test/test_code.txt', 'r') as testcode_file:
#     testcode = testcode_file.read()

# exec testcode

if __name__ == "__main__":
    main()
