"""
    The file reader.
    
    ----------------------------------------------------------------------------
    The MIT License (MIT)
    
    Copyright (c) 2017 Alexandra Instituttet, Aarhus, Danmark
    
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
    ----------------------------------------------------------------------------
    
    """
import numpy as np
import os
import scipy.io as sio


class input_reader_multi(object):

    def __init__(self, load_dir):
        self.internal_counter = 0
        self.cur_dir = load_dir
        self.all_samples = []
        self.curr_frame_idx = 1
        self.find_files(self.cur_dir + '/seq_%s' % str(self.curr_frame_idx).zfill(4))
   
    def find_files(self, load_dir):
        
        for k in range(5):
            j = 1
            for i in range(64):
                data = sio.loadmat(load_dir + '/elem_data_em%s.mat' % str(j+k).zfill(4))
                j += 5
                samples = np.array(data['samples'])
                samples = samples.astype(np.float64)
                mean_samples = np.mean(samples, axis=0, keepdims=True)
                samples = samples - np.tile(mean_samples, [np.shape(samples)[0], 1])
                self.all_samples.append(samples)
        self.all_samples = np.reshape(np.array(self.all_samples), (5, 64, np.shape(self.all_samples)[1],np.shape(self.all_samples)[2]))

    def get_next_batch(self, time_idx=0):
        if self.curr_frame_idx != (time_idx+1):
            self.all_samples = []
            self.find_files(self.cur_dir + '/seq_%s' % str(time_idx+1).zfill(4))
            self.curr_frame_idx = time_idx+1
            self.internal_counter = 0
        samples = self.all_samples[self.internal_counter]
        self.internal_counter = (self.internal_counter + 1) % 5
        return samples
