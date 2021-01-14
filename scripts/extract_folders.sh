#!/bin/bash
list_file=$1
root_dir=$2
output_dir=$3

while read -r pname
do
    echo "inferring folder $root_dir/$pname ..."
    mkdir -p $output_dir/$pname
    CUDA_VISIBLE_DEVICES=0 python run/extract.py $root_dir/$pname $output_dir/$pname 0.1
done < $list_file