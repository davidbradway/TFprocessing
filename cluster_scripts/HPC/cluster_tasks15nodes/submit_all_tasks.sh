#!/bin/bash

task_filename=tf_task            
no_tasks=30   # 1 main task and 1 aux task per node (15 nodes), plus a I/O task in node 0

for (( i=0; i<=${no_tasks}; i++ )); do
      eval "bsub < $task_filename$i"
done

