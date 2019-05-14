# -*- coding: utf-8 -*-
"""
The queue classes serve as mechanism of distributing data/tasks among the
processing tasks servers.

The queue name is shared among all task servers and data is transferred from
one to another when requested.
The current queue implementation does not have a mechanism to minimize the number 
of transfers, as tasks are distributed in a FIFO manner.

In a further implementation a smart queue could attempt to minimize transfer by 
assigning a task (prioritizing them) to a task server that already contain the 
data required.

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
import tensorflow as tf
import numpy as np
import threading
import time

## Input queue that puts data num_epochs times in queue in background thread ##
class InputQueue(object):

    def __init__(self, session, input_queue_size, data_type, read_size, share_name):
        self.data = tf.placeholder(dtype=data_type)
        self.input_queue = tf.FIFOQueue(input_queue_size, data_type, shared_name=share_name)
        self.dequeue_op = self.input_queue.dequeue()
        self.enqueue_op = self.input_queue.enqueue([self.data])
        self.size_op = self.input_queue.size()
        self.close_op = self.input_queue.close()
        self.session = session
        self.read_size = read_size
        self.output = []
        self.closed = False
        self.lockqueue = tf.FIFOQueue(1, tf.int64, shared_name="lock" + share_name)
        self.close_lock_op = self.lockqueue.close()
        self.lock_enqueue = self.lockqueue.enqueue([1])
        self.lock_dequeue = self.lockqueue.dequeue()
        

    def close(self):
        with self.session.graph.as_default():
          self.session.run(self.close_op)
          self.session.run(self.close_lock_op)
          #self.closed = True

    def is_closed(self):
        return self.closed

    def get_read_size(self):
        return self.read_size

    def get_queue(self):
        return self.input_queue

    def get_queue_size(self):
        return self.session.run(self.size_op)

    def lock_queue(self):
        self.session.run(self.lock_enqueue)

    def unlock_queue(self):
        self.session.run(self.lock_dequeue)

    def read_from_queue(self):
        with self.session.graph.as_default():
            #while self.get_queue_size() < self.read_size:
            #    if self.closed:
            #        break
            #    print "looping"
            #    print self.closed
            #    time.sleep(0.1)
            output_list = []
            for i in range(self.read_size):
                output_list.append(self.session.run((self.dequeue_op)))
            if self.read_size == 1:
                self.output = np.asarray(output_list[0])                
            else:    
                self.output = np.asarray(output_list)
            return self.output
            
    #def get_samples(self):
    #    return self.output
           
    def put_in_queue(self, samples, num_epochs=1):
	print "putting in inputqueue"
        with self.session.graph.as_default():
            cnt = 0
            while cnt < num_epochs:
                for i in range(len(samples)):
                    self.session.run(self.enqueue_op, feed_dict={self.data: samples[i]})
                cnt += 1
            print "ending input thread with %d epochs" % cnt
## Local queue ##
class LocalQueue(object):
    def __init__(self, session, queue_size, read_size, data_type, reload_count, share_name, src_queue):
        self.session = session
        self.queue_size = queue_size
        self.input_queue = [None]*queue_size
        self.reload_count = reload_count * queue_size
        self.number_of_loads = 0
        self.src_queue = src_queue
        self.shared_queue = IOQueue(session, queue_size, data_type, read_size, share_name)

    def close(self):
        self.shared_queue.close()

    def read_from_queue(self, index):
        local_index = index % self.queue_size
        if index / self.reload_count > self.number_of_loads:
            del self.input_queue[:]
            self.input_queue = [None] * self.queue_size
            self.number_of_loads += 1
        print index

        if index % self.reload_count == 0:
            print "reload"
            self.src_queue.lock_queue()
            self.shared_queue.lock_queue()
            while self.shared_queue.get_queue_size() > 0:
                self.shared_queue.read_from_queue()
            for i in range(self.queue_size):
                input_sample = self.src_queue.read_from_queue()
                self.shared_queue.put_in_queue([input_sample])
            self.shared_queue.unlock_queue()
            self.src_queue.unlock_queue()

        if self.input_queue[local_index] is None:
            print "fill input queue"
            self.shared_queue.lock_queue()
            # Cycle the whole queue to keep it in the right order
            for i in range(self.queue_size):
                gnaf = self.shared_queue.read_from_queue()
                self.input_queue[i] = gnaf
                self.shared_queue.put_in_queue([gnaf])
            self.shared_queue.unlock_queue()
        output = self.input_queue[local_index]
        return output




## Input/output queue ##
class IOQueue(object):

    def __init__(self, session, input_queue_size, data_type, read_size, share_name):
        with session.graph.as_default():
            self.input_queue = tf.FIFOQueue(input_queue_size, data_type, shared_name=share_name)
            self.data = tf.placeholder(data_type)
            self.dequeue_op = self.input_queue.dequeue()
            self.enqueue_many = self.input_queue.enqueue_many((self.data))
            self.size_op = self.input_queue.size()
            self.close_op = self.input_queue.close()
            self.session = session
            self.read_size = read_size
            self.output = []
            self.closed = False
            self.lockqueue = tf.FIFOQueue(1, tf.int64, shared_name="lock" + share_name)
            self.close_lock_op = self.lockqueue.close()
            self.lock_enqueue = self.lockqueue.enqueue([1])
            self.lock_dequeue = self.lockqueue.dequeue()


    #def set_output_queue(self, output_queue):
        #self.output_queue = output_queue

    def close(self):
        self.session.run(self.close_op)
        self.session.run(self.close_lock_op)
        self.closed = True


    def get_read_size(self):
        return self.read_size

    def get_queue(self):
        return self.input_queue

    def get_queue_size(self):
        return self.session.run(self.size_op)

    def lock_queue(self):
        self.session.run(self.lock_enqueue)

    def unlock_queue(self):
        self.session.run(self.lock_dequeue)

    def read_from_queue(self):
        with self.session.graph.as_default():
            while self.get_queue_size() < self.read_size:
                if self.closed:
                    break
                time.sleep(0.1)
            #output_list = [None]*self.read_size
            #for i in range(self.read_size):
            #    output_list[i] = self.session.run(self.dequeue_op)
            self.output = self.session.run(self.dequeue_op)#output_list[0]
            #if self.output.ndim > 1:
            #    self.output = np.squeeze(self.output)
            return self.output

    #def get_samples(self):
    #    return self.output

            
    def put_in_queue(self, queue_samples, num_epochs = 1):
        with self.session.graph.as_default():
            feed_dict = { self.data : queue_samples }
            cnt = 0
            while cnt < num_epochs:
                self.session.run(self.enqueue_many, feed_dict)
                cnt += 1


## Queue to collect tensors from input/operation queue
class OutputQueue(object):
    
    def __init__(self, session, queue_size, data_type, share_name):
        self.input_queue = tf.FIFOQueue(queue_size, data_type, shared_name=share_name)
        self.data = tf.placeholder(data_type)
        self.enqueue_many = self.input_queue.enqueue_many(self.data)
        self.dequeue = self.input_queue.dequeue()
        self.size_op = self.input_queue.size()
        self.session = session

    def put_in_queue(self, queue_samples):
        with self.session.graph.as_default():
            feed_dict = { self.data : queue_samples }
            self.session.run(self.enqueue_many, feed_dict)

    def read_from_queue(self):
        with self.session.graph.as_default():
            value = self.session.run(self.dequeue)
            return value

    def get_queue(self):
        return self.input_queue

    def get_queue_size(self):
        return self.size_op


class TokenQueue(object):
    def __init__(self, session, queue_size, data_type, share_name):
        self.session = session
        self.data = tf.placeholder(data_type)
        self.queue = tf.FIFOQueue(queue_size, data_type, shared_name=share_name)
        self.enqueue_op = self.queue.enqueue(self.data)
        self.dequeue_op = self.queue.dequeue()
        self.close_op = self.queue.close()

    def read_from_queue_session(self):
        return self.dequeue_op

    def set_feed_session(self, value):
        feed_dict = { self.data : value }
        return feed_dict

    def put_in_queue_session(self):
        return self.enqueue_op

    def read_from_queue(self):
        with self.session.graph.as_default():
            value = self.session.run(self.dequeue_op)
            return value
    
    def put_in_queue(self, value):
        with self.session.graph.as_default():
            feed_dict = { self.data : value }
            self.session.run(self.enqueue_op, feed_dict)

    def close(self):
        self.session.run(self.close_op)
            

class TaskQueue(object):            
    def __init__(self, session, queue_size, data_type, share_name, ordered):
        self.session = session
        self.priority = tf.placeholder(data_type)
        self.data = tf.placeholder(data_type)
        if ordered:
            self.queue = tf.PriorityQueue(queue_size, (data_type), (()), shared_name=share_name)
            self.enqueue_op = self.queue.enqueue((self.priority, self.data))
        else:
            print "creating fifo"
            self.queue = tf.FIFOQueue(queue_size, (data_type, data_type), shared_name=share_name)
            self.enqueue_op = self.queue.enqueue((self.priority, self.data))
        self.val1 = tf.placeholder(tf.int64)
        self.val2 = tf.placeholder(tf.int64)
        self.compare_op = tf.equal(self.val1, self.val2)

        self.dequeue_op = self.queue.dequeue()
        self.close_op = self.queue.close()
        self.ordered = ordered


    def read_from_queue(self, token_value):
        with self.session.graph.as_default():
            value = self.session.run(self.dequeue_op)
            if not self.ordered:
                print "returning not order: " + str(value[0]) + ", " + str(value[1])
                return value[0], value[1]
            feed_dict = { self.val1: token_value, self.val2 : value[0] }
            if self.session.run(self.compare_op, feed_dict):
                return value[0], value[1]
            else:
                feed_dict = { self.priority: value[0], self.data : value[1] }
                self.session.run(self.enqueue_op, feed_dict)
                return value[0], -1
    
    def put_in_queue(self, priority, value):
        with self.session.graph.as_default():
            feed_dict = { self.priority: priority, self.data : value }
            self.session.run(self.enqueue_op, feed_dict)

    def close(self):
        self.session.run(self.close_op)



def queue_reader(token_queue, task_queue, queue, iterations):
    output = []
    token_output = []
    token = token_queue.read_from_queue()
    for i in range(iterations):
        queue_token, task = task_queue.read_from_queue(token)
        while task == -1:
            time.sleep(0.2)
            queue_token, task = task_queue.read_from_queue(token)
        print "read from token, queue_token, task:"
        print token, queue_token, task
        token_output.append(queue_token)
        queue[task].lock_queue()
        output.append(queue[task].read_from_queue())
        queue[task].unlock_queue()
        token = token + 1
    token_queue.put_in_queue(token)
    if iterations == 1:
        return token_output[0], np.asarray(output[0])
    return -1, np.asarray(output)
