#!/bin/bash

rm config/*
uniq $OAR_NODEFILE | python3 config_generator.py


for filename in config/*; do
    IP=$(echo $filename | cut -c18- | tr -d '\n')
    
    oarsh $IP python3 Garfield_TF/Main.py \
        --config Garfield_TF/$filename \
        --log True         \
        --max_iter 2000    \
        --batch_size 64    \
        --dataset cifar10  \
        --nbbyzwrks 3      \
        --model VGG &
done