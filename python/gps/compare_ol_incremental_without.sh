#!/bin/bash

for ((i=2;i<=5;i++))
do
    dir="position/compare_alpha_"$i
    # mkdir $dir
    # python python/gps/compare_ol_alpha.py mjc_olgps_example
    python python/gps/compare_ol_without_alpha.py mjc_olgps_example
    mv position/*.pkl $dir
done