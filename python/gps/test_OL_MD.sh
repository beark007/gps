#!/bin/bash

for ((i=1;i<=3;i++))
do
    dir="position/md_ol_"$i
    echo $dir
    mkdir $dir
    python python/gps/generate_pos.py -m 1
    python python/gps/olgps_explain.py mjc_olgps_example
    python python/gps/mdgps_explain.py mjc_mdgps_example
    mv position/*.pkl $dir
done