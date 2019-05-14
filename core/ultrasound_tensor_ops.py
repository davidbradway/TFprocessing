"""
    The UltrasounOP Class contains the operations (nodes) used for processing the
    data.
    The operations can be expressed using native tensor ops or by using custom ops 
    loaded from a dll, or a combination of both. 
    
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader
import numpy as np
import tensorflow as tf
from decimal import *


import os
cur_dir = os.getcwd()

class UltrasoundOp:
  def __init__(self):
    self._ultrasound_op = load_library.load_op_library('/src/core/ultrasound.so')
    assert self._ultrasound_op, "Could not load ultrasound.so."

  def add_one(self, inarray):
    return self._ultrasound_op.add_one(inarray)


  def polar_grid(self, lambda_val, line_length, d_line, angles_theta, angles_phi, estimation_point):
    return self._ultrasound_op.polar_grid(lambda_val, line_length, d_line, angles_theta, angles_phi, estimation_point)


  def time_of_flight(self, focus, center_focus, speed_of_sound, t_start, element_pos, point_pos):
    return self._ultrasound_op.time_of_flight(focus, center_focus, speed_of_sound, t_start, element_pos, point_pos)

  def linear_interpolation(self, t_start, f_sampling, tof_per_bf_point, samples):
    #samples = tf.cond(3 > tf.rank(samples), lambda: tf.pad(samples, [[1,1],[0,0]], "CONSTANT"), lambda: tf.pad(samples, [[0,0],[1,1],[0,0]], "CONSTANT"))
    samples = tf.pad(samples, [[0,0],[1,1],[0,0]], "CONSTANT")
    return self._ultrasound_op.linear_interpolation(t_start, f_sampling, tof_per_bf_point, samples)



  def none_and_sum_apod(self, delayed_samples):
    return self._ultrasound_op.none_and_sum_apod(delayed_samples)
    #return tf.reduce_sum((delayed_samples), 1)


  def dynamic_and_sum_apod(self, f_number, element_pos, bf_point, delayed_samples):
    return self._ultrasound_op.dynamic_and_sum_apod(f_number, element_pos, bf_point, delayed_samples)

  def beamform(self, focus, center_focus, speed_of_sound, t_start, t_start_data, f_sampling, f_number, element_pos, point_pos, samples):
    samples = tf.pad(samples, [[0,0],[1,1],[0,0]], "CONSTANT")
    return self._ultrasound_op.beamforming(focus, center_focus, speed_of_sound, t_start, t_start_data, f_sampling, f_number, element_pos, point_pos, samples)

  def beamform_avx(self, focus, center_focus, speed_of_sound, t_start, t_start_data, f_sampling, f_number, element_pos, point_pos, samples):
    samples = tf.pad(samples, [[0,0],[1,1],[0,0]], "CONSTANT")
    return self._ultrasound_op.beamforming_avx(focus, center_focus, speed_of_sound, t_start, t_start_data, f_sampling, f_number, element_pos, point_pos, samples)


  def echo_cancel(self, bf_samples):
    return self._ultrasound_op.echo_cancel(bf_samples)

  def echo_cancel_threshold(self, bf_samples, fourier_threshold, tukey_ec, tukey_ec_freq, percentage):
    return self._ultrasound_op.echo_cancel_threshold(bf_samples, fourier_threshold, tukey_ec, tukey_ec_freq, percentage)

  def cart_to_polar(self, point_dt, lambda_val, line_length, d_line, angles_theta, points, grid_dimensions, samples, all_angles):
    return self._ultrasound_op.cart_to_polar(point_dt, lambda_val, line_length, d_line, angles_theta, points, grid_dimensions, samples, all_angles)

  def tukeywin(self, window_length, alpha=0.5):
    '''The Tukey window, also known as the tapered cosine window, can be regarded as a cosine lobe of width \alpha * N / 2
    that is convolved with a rectangle window of width (1 - \alpha / 2). At \alpha = 1 it becomes rectangular, and
    at \alpha = 0 it becomes a Hann window.
 
    We use the same reference as MATLAB to provide the same results in case users compare a MATLAB output to this function
    output
 
    Reference
    ---------
    http://www.mathworks.com/access/helpdesk/help/toolbox/signal/tukeywin.html
 
    '''
    # Special cases
    if alpha <= 0:
        return np.ones(window_length) #rectangular window
    elif alpha >= 1:
        return np.hanning(window_length)
 
    # Normal case
    x = np.linspace(0, 1, window_length)
    w = np.ones(x.shape)
 
    # first condition 0 <= x < alpha/2
    first_condition = x<alpha/2
    w[first_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[first_condition] - alpha/2) ))
 
    # second condition already taken care of
 
    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = x>=(1 - alpha/2)
    w[third_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[third_condition] - 1 + alpha/2))) 
 
    return w

  def parabolic_x_corr(self, lambda_val, t_prf_eff, line_length, d_line, samples, tukey):
    return self._ultrasound_op.parabolic_x_corr(lambda_val, t_prf_eff, line_length, d_line, samples, tukey)

  def mcd3d(self, samples, angles_theta, angles_phi):
    return self._ultrasound_op.mcd3d(samples, angles_theta, angles_phi)
