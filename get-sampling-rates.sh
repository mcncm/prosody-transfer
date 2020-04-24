#!/bin/bash

for f in Blizzard-Challenge-2013/*
do
    path=`echo -n $f`
    echo -n $(basename $path) >> sampling-rates
    echo -n " " >> sampling-rates
    soxi -r $path >> sampling-rates
done;

