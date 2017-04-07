#!/bin/bash

for ((i=1;i<=3;i++))
do
    dir="position/ol_ability_"$i
    mkdir $dir
    python python/gps/generate_pos.py
    python python/gps/ol_learn_ability.py mjc_olgps_example
    mv position/*.pkl $dir
    cp $dir/position_train.pkl position/.
done

for ((j=1;j<=2;j++))
do
    dir="position/compare_alpha_"$j
    mkdir $dir
    python python/gps/compare_ol_alpha.py mjc_olgps_example
    python python/gps/compare_ol_without_alpha.py mjc_olgps_example
    mv *.pkl $dir
done

    #dir="position/ol_ability_"$i
    #mkdir $dir
    #python python/gps/ol_learn_ability.py mjc_olgps_example
    #mv position/*.pkl $dir
    #cp $dir/position_train.pkl position/.