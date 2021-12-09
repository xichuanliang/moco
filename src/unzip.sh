#!/bin/bash

#dir=/home/duhan/codes/dataset/ILSVRC2012数据集/train
#for x in `ls $dir/*tar` do
#  filename=`basename $x .tar`
#  mkdir $dir/$filename
#  tar -xvf $x -C $dir/$filename
#done
#rm *.tar

python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=2 --use_env main.py