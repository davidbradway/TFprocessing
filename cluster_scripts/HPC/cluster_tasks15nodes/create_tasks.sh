#!/bin/bash

# Nodes to use
queuename=cfu                 # Queue to use
nodebasename=n-62-21-             # Base name for the nodes 
ipbaseaddr=10.66.21.              # IP base address of the nodes 
timelimit=24                      # timelimit in hours
proc_nodes=({53..60} 62 63 64 65 66 67 68)  # Processing nodes

task_filename=tf_task             # Task filename     

# IO JOB settings
io_node=52                        # IO node for task 0 
memusage_IO=100000                # IO memusage [MB]
cpuusage_IO=24                    # CPU memusage [cores]
port_IO=2500                      # TCP port

# Main job settings
memusage_main=22000             # IO memusage  [MB]
cpuusage_main=22                # CPU memusage [cores]
port_main=2500                  # TCP port

# Aux job settings
memusage_aux=12000             # IO memusage  [MB]
cpuusage_aux=2                 # CPU memusage [cores]
port_aux=4500                  # TCP port


# Create worker hosts string 
workers_string=$ipbaseaddr
workers_string+=$io_node
workers_string+=":"
workers_string+=$port_IO
# get length of an array
tLen=${#proc_nodes[@]}
# Generate main job 
for (( i=0; i<${tLen}; i++ )); do
    workers_string+=,
    workers_string+=$ipbaseaddr
    workers_string+=${proc_nodes[$i]}
    workers_string+=":"
    workers_string+=$port_main
done
# Generate aux job 
for (( i=0; i<${tLen}; i++ )); do
    workers_string+=,
    workers_string+=$ipbaseaddr
    workers_string+=${proc_nodes[$i]}
    workers_string+=":"
    workers_string+=$port_aux
done

# Create IO task, task 0
temp=$task_filename 
temp+=0
if [ -e $temp ]; then
  echo "File $temp already exists!"
else
  echo "#!/bin/sh"  >> $temp
  echo "#BSUB -J tensorflow_grpc" >> $temp
  echo "#BSUB -o logs/%J.log" >> $temp
  echo "#BSUB -m \"$nodebasename$io_node\"" >> $temp
  echo "#BSUB -W $timelimit:00" >> $temp
  echo "#BSUB -R \"rusage[mem=$((memusage_IO / cpuusage_IO))]\"" >> $temp
  echo "#BSUB -n $cpuusage_IO" >> $temp
  echo "#BSUB -q $queuename" >> $temp
  echo " " >> $temp
  echo "OMP_NUM_THREADS=\$LSB_DJOB_NUMPROC" >> $temp 
  echo "export OMP_NUM_THREADS" >> $temp 
  echo "module load python/2.7.12_ucs4" >> $temp
  echo "module load h5py/2.7.1-python-2.7.12_ucs4" >> $temp
  echo "module load numpy/1.13.3-python-2.7.12_ucs4-openblas-0.2.20" >> $temp
  echo "module load scipy/0.19.1-numpy-1.13.3-python-2.7.12_ucs4" >> $temp
  echo "source /appl/tensorflow/1.1cpu-python2712/bin/activate" >> $temp
  echo "python /src/vfi_cluster.py --worker_hosts=$workers_string --job_name=worker --task_index=0" >> $temp
fi

# Create main tasks
for (( i=0; i<${tLen}; i++ )); do
  temp=$task_filename 
  temp+=$((i+1))
  if [ -e $temp ]; then
  	echo "File $temp already exists!"
  else
  	echo "#!/bin/sh"  >> $temp
  	echo "#BSUB -J tensorflow_grpc" >> $temp
  	echo "#BSUB -o logs/%J.log" >> $temp
  	echo "#BSUB -m \"$nodebasename${proc_nodes[$i]}\"" >> $temp
  	echo "#BSUB -W $timelimit:00" >> $temp
  	echo "#BSUB -R \"rusage[mem=$((memusage_main / cpuusage_main))]\"" >> $temp
  	echo "#BSUB -n $cpuusage_main" >> $temp
  	echo "#BSUB -q $queuename" >> $temp
  	echo " " >> $temp
  	echo "OMP_NUM_THREADS=\$LSB_DJOB_NUMPROC" >> $temp 
  	echo "export OMP_NUM_THREADS" >> $temp 
  	echo "module load python/2.7.12_ucs4" >> $temp
  	echo "module load h5py/2.7.1-python-2.7.12_ucs4" >> $temp
  	echo "module load numpy/1.13.3-python-2.7.12_ucs4-openblas-0.2.20" >> $temp
  	echo "module load scipy/0.19.1-numpy-1.13.3-python-2.7.12_ucs4" >> $temp
  	echo "source /appl/tensorflow/1.1cpu-python2712/bin/activate" >> $temp
  	echo "python /src/vfi_cluster.py --worker_hosts=$workers_string --job_name=worker --task_index=$((i+1))" >> $temp
  fi 
done	

# Create aux tasks
for (( i=0; i<${tLen}; i++ )); do
  temp=$task_filename 
  temp+=$((i+1+tLen))
  if [ -e $temp ]; then
  	echo "File $temp already exists!"
  else
  	echo "#!/bin/sh"  >> $temp
  	echo "#BSUB -J tensorflow_grpc" >> $temp
  	echo "#BSUB -o logs/%J.log" >> $temp
  	echo "#BSUB -m \"$nodebasename${proc_nodes[$i]}\"" >> $temp
  	echo "#BSUB -W $timelimit:00" >> $temp
  	echo "#BSUB -R \"rusage[mem=$((memusage_aux / cpuusage_aux))]\"" >> $temp
  	echo "#BSUB -n $cpuusage_aux" >> $temp
  	echo "#BSUB -q $queuename" >> $temp
  	echo " " >> $temp
  	echo "OMP_NUM_THREADS=\$LSB_DJOB_NUMPROC" >> $temp 
  	echo "export OMP_NUM_THREADS" >> $temp 
  	echo "module load python/2.7.12_ucs4" >> $temp
  	echo "module load h5py/2.7.1-python-2.7.12_ucs4" >> $temp
  	echo "module load numpy/1.13.3-python-2.7.12_ucs4-openblas-0.2.20" >> $temp
  	echo "module load scipy/0.19.1-numpy-1.13.3-python-2.7.12_ucs4" >> $temp
  	echo "source /appl/tensorflow/1.1cpu-python2712/bin/activate" >> $temp
  	echo "python /src/vfi_cluster.py --worker_hosts=$workers_string --job_name=worker --task_index=$((i+1+tLen))" >> $temp
  fi 
done	
