# -*- coding: utf-8 -*-
"""
The TParser reads the xml configuration file and generate the code for creating
the tensorflow graph to be executed. It creates the queues and connects them 
together.



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
import xml.etree.ElementTree as et
import re

class TParser(object):

    def __init__(self, path):
        new_file = open(path, "r")
        xml_string = new_file.read()
        xml_string = xml_string.replace("<code>", "<code>\n<![CDATA[")
        xml_string = xml_string.replace("</code>", "]]>\n</code>")
        xml_string = re.sub("(<loop.*?>)", r"]]>\n</code>\n<code>\n\1\n<![CDATA[", xml_string)
        xml_string = xml_string.replace("</loop>", "]]>\n</loop>\n</code>\n<code>\n<![CDATA[")
        

        #print xml_string
        tree = et.fromstring(xml_string)
        self.root = tree#.getroot()
        self.commandline = ""
        self.indentation = 0
        self.queue_type = {}
        self.queue_data_type = {}
        self.queue_read_size = {}
        self.param_queue_names = []
        self.inout_queue_names = []
        self.local_queue_names = []
        self.inout_ordered = {}
        self.out_queue_names = []
        self.token_queue_names = []
        self.token_method_queue_names = {}
        self.task_queue_names = []
        
        self.no_of_tasks = 0

    def indent(self):
        text = "\n"
        for i in range(self.indentation):
            text += "  "
        return text

    def parse_xml(self):
        self.create_tasks()
        self.create_input_output_queues()
        self.create_inout_queues()
        self.create_local_queues()
        self.create_task_queues()
        self.write_param_queues()
        self.write_close_queues()
        self.write_get_out_queues()
        self.create_methods()
        self.create_input_method()
        self.create_output_method()
        self.init_token_queues()
        self.init_session()
        self.write_main_loop()




    def create_input_method(self):
        feeder = self.root.find('input_method')
        name = feeder.get('name')
        self.commandline += self.indent() + "def " + name + "():"
        self.indentation += 1
        for code in feeder.findall('code'):
            text = code.text
            start_indent = -1
            for t in text.split('\n'):
                if t.strip() != '':
                    if start_indent == -1:
                        start_indent = len(t) - len(t.lstrip())
                    self.indentation += len(t) - len(t.lstrip()) - start_indent
                    self.commandline += self.indent() + t.strip()
                    self.indentation -= len(t) - len(t.lstrip()) - start_indent
        self.indentation -= 1
        self.commandline += "\n"

    def create_output_method(self):
        end_loop = self.root.find('output_method')
        name = end_loop.get('name')
        self.commandline += self.indent() + "def " + name + "():"
        self.indentation += 1
        for code in end_loop.findall('code'):
            text = code.text
            start_indent = -1
            for t in text.split('\n'):
                if t.strip() != '':
                    if start_indent == -1:
                        start_indent = len(t) - len(t.lstrip())
                    self.indentation += len(t) - len(t.lstrip()) - start_indent
                    self.commandline += self.indent() + t.strip()
                    self.indentation -= len(t) - len(t.lstrip()) - start_indent
        self.indentation -= 1
        self.commandline += "\n"



    def create_tasks(self):
        tasks = str(int(self.root.find('tasks').text) + 1)
        self.no_of_tasks = tasks
        task_methods = self.root.find('task_methods').text
        self.commandline += self.indent() + "task_numbers = " + tasks
        self.commandline += self.indent() + "task_methods = '" + task_methods + "'"
        self.commandline += self.indent() + "task_method_list = [x.strip() for x in task_methods.split(',')]"
        self.commandline += self.indent() + "initial_tasks = {}"
        self.commandline += self.indent() + "for i in range(1, task_numbers):"

        self.indentation += 1
        self.commandline += self.indent() + "initial_tasks[i] = task_method_list[i-1]"
        self.indentation -= 1
        self.commandline += "\n\n"




    def create_methods(self):
        self.commandline += "\n\nultra_op = UltrasoundOp()\n"  
        for method in self.root.findall('method'):
            method_name = method.get('name')
            placeholder_dict = {}
            graph_code_dict = {}

            self.commandline += "\n\nwith tf.device('job:%s/task:%d/CPU' % (FLAGS.job_name, FLAGS.task_index)):"
            self.indentation += 1


            for code in method.findall('code'):
                for loops in code.findall('loop'):
                    loop_input = loops.get('input')
                    loop_par = loops.get('parallel')
                    loop_iter = loops.get('iterations')
                    loop_output = loops.get('output')
                    text = loops.text

                    self.commandline += self.indent() + "def " + method_name + "_loop_cond(__OutArr, " + loop_input + ", i):"
                    self.indentation += 1
                    self.commandline += self.indent() + "return i < " + loop_iter + "\n"
                    self.indentation -= 1

                    self.commandline += self.indent() + "def " + method_name + "_loop_body(__OutArr, " + loop_input + ", i):"
                    self.indentation += 1
                    start_indent = -1
                    for t in text.split('\n'):
                        if t.strip() != '':
                            if start_indent == -1:
                                start_indent = len(t) - len(t.lstrip())
                            self.indentation += len(t) - len(t.lstrip()) - start_indent
                            self.commandline += self.indent() + t.strip()
                            self.indentation -= len(t) - len(t.lstrip()) - start_indent

                    self.commandline += self.indent() + "__OutArr = __OutArr.write(i, " + loop_output + ")"
                    self.commandline += self.indent() + "return (__OutArr, " + loop_input + ", i+1)"
                    self.indentation -= 1

            self.commandline += "\n" + self.indent() + "def " + method_name + "_graph_code():"
            self.indentation += 1 

            self.commandline += self.indent() + "_placeholder_dict = {}"
            for feed_data in method.findall('feed_arr'):
                name = feed_data.get('name')
                input_data_type = feed_data.get('data_type')
                if input_data_type == "float":
                    input_data_type = "tf.float32"
                elif input_data_type == "double":
                    input_data_type = "tf.float64"
                elif input_data_type == "uint16":
                    input_data_type = "tf.uint16"
                elif input_data_type == "complex128":
                    input_data_type = "tf.complex128"
                self.commandline += self.indent() + name + " = tf.placeholder(" + input_data_type + ", name = '" + name + "')"
                self.commandline += self.indent() + "_placeholder_dict['" + name + "'] = " + name


            for input_data in method.findall('input_data'):
                name = input_data.get('name')
                queue = input_data.get('queue')
                input_data_type = self.queue_data_type[queue]
                if input_data_type == "float":
                    input_data_type = "tf.float32"
                elif input_data_type == "double":
                    input_data_type = "tf.float64"
                elif input_data_type == "uint16":
                    input_data_type = "tf.uint16"
                elif input_data_type == "complex128":
                    input_data_type = "tf.complex128"
                self.commandline += self.indent() + name + " = tf.placeholder(" + input_data_type + ", name = '" + name + "')"
                self.commandline += self.indent() + "_placeholder_dict['" + name + "'] = " + name


            for code in method.findall('code'):
                text = code.text

                start_indent = -1
                for t in text.split('\n'):
                    if t.strip() != '':
                        if start_indent == -1:
                            start_indent = len(t) - len(t.lstrip())
                        self.indentation += len(t) - len(t.lstrip()) - start_indent
                        self.commandline += self.indent() + t.strip()
                        self.indentation -= len(t) - len(t.lstrip()) - start_indent
                for loops in code.findall('loop'):
                    loop_input = loops.get('input')
                    loop_par = loops.get('parallel')
                    loop_iter = loops.get('iterations')
                    loop_output = loops.get('output')
                    self.commandline += self.indent() + "__OutArr = tf.TensorArray(tf.float64, " + loop_iter + ")"
                    self.commandline += self.indent() + "__OutArr = tf.while_loop(" + method_name + "_loop_cond, " + method_name + "_loop_body, [__OutArr, " + loop_input + ", 0], parallel_iterations = " + loop_par + ")[0]"
                    self.commandline += self.indent() + loop_output + "= tf.unstack(__OutArr.stack(), " + loop_iter + ")"


            self.commandline += self.indent() + "graph_code_dict = {}"
            for returns in method.findall('output_data'):
                ultra_op = returns.get('value')
                self.commandline += self.indent() + "graph_code_dict['" + ultra_op + "'] = " + ultra_op

            self.commandline += self.indent() + "return _placeholder_dict, graph_code_dict"

            self.indentation -= 1
            self.commandline += "\n" + self.indent() + method_name + "_placeholder_dict, " + method_name + "_graph_dict = " + method_name + "_graph_code()" 
            self.indentation -= 1
            self.commandline += "\n"


            method_token = method.get('token_name')
            if method_token == None:
                method_token = method_name
            self.commandline += self.indent() + "def " + method_name + "_method():\n"
            self.indentation += 1
            self.commandline += self.indent() + "token = " + method_token + "_method_token.read_from_queue()\n"
            self.commandline += self.indent() + "feed_dict = {}"

            for feed_data in method.findall('feed_arr'):
                name = feed_data.get('name')
                variable = feed_data.get('variable')
                indexer = feed_data.get('index')
                if indexer is None:
                    self.commandline += self.indent() + "feed_dict.update({ " + method_name + "_placeholder_dict['"+ name + "'] : " + variable + " })"
                else:
                    self.commandline += self.indent() + "feed_dict.update({ " + method_name + "_placeholder_dict['"+ name + "'] : " + variable + "[" + indexer + "] })"

            for input_data in method.findall('input_data'):
                name = input_data.get('name')
                queue = input_data.get('queue')
                input_data_type = self.queue_data_type[queue]
                read_size = self.queue_read_size[queue]
                if input_data_type == "float":
                    input_data_type = "tf.float32"
                elif input_data_type == "double":
                    input_data_type = "tf.float64"
                elif input_data_type == "uint16":
                    input_data_type = "tf.uint16"
                elif input_data_type == "complex128":
                    input_data_type = "tf.complex128"
                #self.commandline += self.indent() + name + " = tf.placeholder(" + input_data_type + ", name = '" + name + "')"
                if self.queue_type[queue] == "inout":
                    self.commandline += self.indent() + "queue_token, feed_" + name + " = queue_reader(" + queue + "_token, " + queue + "_task, " + queue + "_queues, " + read_size + ")"
                    if(not self.inout_ordered[queue]):
                        self.commandline += self.indent() + "if queue_token != -1:"
                        self.indentation += 1
                        self.commandline += self.indent() + "token = queue_token"
                        self.indentation -= 1
                elif self.queue_type[queue] == "input":
                    self.commandline += self.indent() + queue + "_queue.lock_queue()"
                    self.commandline += self.indent() + "feed_" + name + " = " + queue + "_queue.read_from_queue()"
                    #self.commandline += self.indent() + "feed_" + name + " = " + queue + "_queue.get_samples()"
                    self.commandline += self.indent() + queue + "_queue.unlock_queue()"
                elif self.queue_type[queue] == "local":
                    self.commandline += self.indent() + "feed_" + name + " = " + queue + "_queue.read_from_queue(token)"
                self.commandline += self.indent() + "feed_dict.update({ " + method_name + "_placeholder_dict['" + name + "'] : feed_" + name + " })"
            self.commandline += self.indent() + method_token + "_method_token.put_in_queue(token+1)\n"
            for user_code in method.findall('user_code'):
                self.commandline += self.indent() + user_code.text
#            for code in method.findall('code'):
#                text = code.text
#                start_indent = -1
#                for t in text.split('\n'):
#                    if t.strip() != '':
#                        if start_indent == -1:
#                            start_indent = len(t) - len(t.lstrip())
#                        self.indentation += len(t) - len(t.lstrip()) - start_indent
#                        self.commandline += self.indent() + t.strip()
#                        self.indentation -= len(t) - len(t.lstrip()) - start_indent
            for ultra_op in method.findall('ultra_op'):
                param_name = ultra_op.get('name')
                op_name = ultra_op.get('op_name')
                params = ultra_op.find('params').text
                self.commandline += self.indent() + "" + param_name + " = " + op_name + "(" + params + ")"
            self.commandline += self.indent()
            for returns in method.findall('output_data'):
                ultra_op = returns.get('value')
                queue = returns.get('queue_name')
                if self.queue_type[queue] == "inout":
                    repeat = returns.get('repeat')
                    if repeat == None:
                        repeat = "1"
                    self.commandline += self.indent() + "for __repeat in range(" + repeat + "):"
                    self.indentation += 1
                    self.commandline += self.indent() + queue + "_task.put_in_queue(token * " + repeat + " + __repeat , FLAGS.task_index)"
                    self.indentation -= 1
                    self.commandline += self.indent() + queue + "_queues[FLAGS.task_index].put_in_queue([session.run(" + method_name + "_graph_dict['" + ultra_op + "'], feed_dict)], " + repeat + ")"
                elif self.queue_type[queue] == "output":
                    self.commandline += self.indent() + queue + "_queue.put_in_queue([session.run(" + method_name + "_graph_dict['" + ultra_op + "'], feed_dict)])"
            self.indentation -= 1
            self.commandline += "\n\n"

        self.commandline += self.indent() + "master_method_array = {}"
        for method in self.root.findall('method'):
            method_name = method.get('name')
            self.commandline += self.indent() + "master_method_array['" + method_name + "'] = " + method_name + "_method"## + "()"

        self.commandline += "\n\n"
    

    def create_input_output_queues(self):
        c = self.root
        self.commandline += "with tf.device('job:%s/task:%d/CPU' % (FLAGS.job_name, 0)):"
        self.indentation += 1
        queues = c.find('queues')
        #Create list of all queues
        for queue in queues.findall('queue'):
            name = queue.get('name')
            queue_type = queue.get('type')
            capacity = queue.get('capacity')
            data_type = queue.get('data_type')
            if queue_type == "output":
                self.commandline += self.indent() + name + "_queue = OutputQueue(session, " + capacity + ", '" + data_type + "', '" + name + "_queue_0')"
                self.queue_type[name] = "output"
                self.out_queue_names.append(name)
            elif queue_type == "input":
                read_size = queue.get('read_size')
                r_size = "1"
                self.commandline += self.indent() + name + "_queue = InputQueue(session, " + capacity + ", '" + data_type + "', " + read_size + ", '" + name + "_queue_0')"
                self.queue_type[name] = "input"
                self.queue_data_type[name] = data_type
                self.queue_read_size[name] = read_size
                self.param_queue_names.append(name)
        self.commandline += "\n\n"
        self.indentation -= 1


    def create_inout_queues(self):
        inout = []
        c = self.root
        queues = c.find('queues')
        #Create list of all queues
        for queue in queues.findall('queue'):
            name = queue.get('name')
            queue_type = queue.get('type')
            if queue_type == "inout":
                self.commandline += self.indent() + name + "_queues = {}"
                self.queue_type[name] = "inout"
                self.inout_queue_names.append(name)
        if(len(self.inout_queue_names) > 0):
            self.commandline += self.indent() + "for i in range(1, task_numbers):"
            self.indentation += 1
            self.commandline += self.indent() + "with tf.device('job:%s/task:%d/CPU' % (FLAGS.job_name, i)):"
            self.indentation += 1
            for queue in queues.findall('queue'):
                name = queue.get('name')
                queue_type = queue.get('type')
                capacity = queue.get('capacity')
                read_size = queue.get('read_size')
                r_size = "1"
                data_type = queue.get('data_type')
                if queue_type == "inout":
                    self.commandline += self.indent() + name + "_queue = IOQueue(session, " + capacity + ", '" + data_type + "', " + r_size + ", '" + name + "_queue_%d' % i)"
                    self.queue_data_type[name] = data_type
                    self.queue_read_size[name] = read_size
                    inout.append(name)
            self.indentation -= 1
            for i in range(len(inout)):
                self.commandline += self.indent() + inout[i] +"_queues[i] = " + inout[i] + "_queue"
            self.commandline += "\n\n"
            self.indentation -= 1


    def create_local_queues(self):
        local = []
        c = self.root
        queues = c.find('queues')
        #Create list of all queues
        for queue in queues.findall('queue'):
            name = queue.get('name')
            queue_type = queue.get('type')
            if queue_type == "local":

                capacity = queue.get('capacity')
                read_size = queue.get('read_size')
                r_size = "1"
                data_type = queue.get('data_type')
                read_before_reload = queue.get('queue_cycles_before_reload')
                src_queue = queue.get('src_queue')

                self.queue_type[name] = "local"
                self.local_queue_names.append(name)
                self.commandline += self.indent() + name + "_queue = LocalQueue(session, " + capacity +  ", " + read_size + ", '" + data_type+ "', " + read_before_reload + ", '" + name + "_queue', " + src_queue + "_queue)"
                self.queue_data_type[name] = data_type
                self.queue_read_size[name] = read_size
        self.commandline += "\n\n"



    def create_task_queues(self):
        c = self.root
        self.commandline += "with tf.device('job:%s/task:%d/CPU' % (FLAGS.job_name, 0)):"
        self.indentation += 1

        for method in c.findall('method'):
            name = method.get('name')
            token = method.get('token_name')
            if token == None:
                token = name
            if not token in self.token_method_queue_names:    
                self.commandline += self.indent() + token + "_method_token = TokenQueue(session, 1, tf.int64, '" + token + "_method_token')"
                self.token_method_queue_names[token] = token + "_method_token"
        queues = c.find('queues')
        for queue in queues.findall('queue'):
            name = queue.get('name')
            queue_type = queue.get('type')
            capacity = queue.get('capacity')
            if queue_type == "inout":
                ordered = queue.get('ordered')
                if ordered == None:
                    ordered = "True"
                self.inout_ordered[name] = ordered == "True"
                self.commandline += self.indent() + name + "_token = TokenQueue(session, 1, tf.int64, '" + name + "_token')"
                self.commandline += self.indent() + name + "_task = TaskQueue(session, " + capacity + ", tf.int64, '" + name + "_task', " + ordered + ")"
                self.token_queue_names.append(name + "_token")
                self.task_queue_names.append(name + "_task")
        self.commandline += "\n\n"
        self.indentation -= 1


    def write_param_queues(self):
        param_queue_names = []
        for i in range(len(self.param_queue_names)):
            param_queue_names.append(self.param_queue_names[i] + "_data")
        queue_names = ",".join(param_queue_names)
        self.commandline += self.indent() + "def put_data_in_queues(" + queue_names + "):"
        self.indentation += 1
        for i in range(len(self.param_queue_names)):
            self.commandline += self.indent() + self.param_queue_names[i] + "_queue.put_in_queue(" + self.param_queue_names[i] + "_data)"
        self.indentation -= 1
        self.commandline += "\n\n"

                                            
    def write_close_queues(self):
        self.commandline += self.indent() + "def close_all_queues():"
        self.indentation += 1
        
        for i in range(len(self.inout_queue_names)):
            self.commandline += self.indent() + self.inout_queue_names[i] + "_queue.close()"
        for i in range(len(self.local_queue_names)):
            self.commandline += self.indent() + self.local_queue_names[i] + "_queue.close()"

        for key in self.token_method_queue_names:
            self.commandline += self.indent() + self.token_method_queue_names[key] + ".close()"
        for i in range(len(self.token_queue_names)):
            self.commandline += self.indent() + self.token_queue_names[i] + ".close()"
        for i in range(len(self.task_queue_names)):
            self.commandline += self.indent() + self.task_queue_names[i] + ".close()"
        for i in range(len(self.param_queue_names)):
            self.commandline += self.indent() + self.param_queue_names[i] + "_queue.close()"
        self.indentation -= 1
        self.commandline += "\n\n"

    def write_get_out_queues(self):
        self.commandline += self.indent() + "def get_out_queues():"
        queue_list = []
        for i in range(len(self.out_queue_names)):
            queue_list.append(self.out_queue_names[i] + "_queue")
        queue_names = ', '.join(map(str, queue_list)) 
        self.indentation += 1
        self.commandline += self.indent() + "return " + queue_names
        self.indentation -= 1
        self.commandline += "\n\n"
        

    def init_token_queues(self):
        self.commandline += self.indent() + "if(FLAGS.task_index == 0):"
        self.indentation += 1
        for key in self.token_method_queue_names:
            self.commandline += self.indent() + self.token_method_queue_names[key] + ".put_in_queue(0)"

        for i in range(len(self.token_queue_names)):
            self.commandline += self.indent() + self.token_queue_names[i] + ".put_in_queue(0)"
        self.indentation -= 1
        self.commandline += "\n\n"

    def init_session(self):
        self.commandline += self.indent() + "end_condition__ = tf.get_variable('end_condition__', [], tf.int64, initializer=tf.constant_initializer(0))"
        self.commandline += self.indent() + "update_end_condition__ = end_condition__.assign_add(1, use_locking=True)"
        self.commandline += self.indent() + "init = tf.global_variables_initializer()\nsession.run(init)\nsession.graph.finalize()"

    def start_queue_runners(self):
        self.commandline += "\ncoord = tf.train.Coordinator()\nthreads = tf.train.start_queue_runners(sess=session)"

    def write_main_loop(self):
        input_method = self.root.find('input_method')
        output_method = self.root.find('output_method')
        input_method_name = input_method.get('name')
        output_method_name = output_method.get('name')
        self.commandline += self.indent() + "def main():"
        self.indentation += 1        
        self.commandline += self.indent() + "with session as sess: "
        self.indentation += 1
        self.commandline += self.indent() + "try:"
        self.indentation += 1
        self.commandline += self.indent() + "while True:"
        self.indentation += 1
        self.commandline += self.indent() + "if(FLAGS.task_index == 0):"
        self.indentation += 1
        self.commandline += self.indent() + "with tf.device('job:%s/task:%d/CPU' % (FLAGS.job_name, FLAGS.task_index)):"
        self.indentation += 1
        self.commandline += self.indent() + "input_worker = threading.Thread(target=" + input_method_name + ")"
        self.commandline += self.indent() + "input_worker.setDaemon(True)"
        self.commandline += self.indent() + "input_worker.start()"
        self.commandline += self.indent() + output_method_name + "()"
        self.commandline += self.indent() + "close_all_queues()"
        self.commandline += self.indent() + "break"
        
        self.indentation -= 2
        self.commandline += self.indent() + "else:"
        self.indentation += 1
        self.commandline += self.indent() + "with tf.device('job:%s/task:%d/CPU' % (FLAGS.job_name, FLAGS.task_index)):"        
        self.indentation += 1
        self.commandline += self.indent() + "tasks = initial_tasks[FLAGS.task_index]"
        self.commandline += self.indent() + "print tasks"
        self.commandline += self.indent() + "for task in tasks.split(';'):"
        self.indentation += 1
        self.commandline += self.indent() + "master_method_array[task]()"
        self.commandline += self.indent() + "print 'running method %s' % master_method_array[task]"
        self.indentation -= 5
        self.commandline += self.indent() + "except (tf.errors.OutOfRangeError, tf.errors.CancelledError):"
        self.indentation += 1
        self.commandline += self.indent() + "print 'closed'"
        self.indentation -= 1
        self.commandline += self.indent() + "session.run(update_end_condition__)"
        self.commandline += self.indent() + "while session.run(end_condition__) < " + self.no_of_tasks + ":"
        self.indentation += 1
        self.commandline += self.indent() + "time.sleep(0.5)"
        self.indentation -= 1
        self.commandline += self.indent() + "print 'Exiting program in 2 seconds'"
        self.commandline += self.indent() + "time.sleep(2)"
        self.indentation -= 1

    def stop_session(self):
        self.commandline += "\ncoord.request_stop()\ncoord.join(threads)\nsession.close()"

    def print_commands(self):
        print self.commandline


    def return_code(self):
        return self.commandline
