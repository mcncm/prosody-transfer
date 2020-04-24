#!/bin/bash

for f in LJSpeech-1.1/wavs/*
do
    path=`echo -n $f`
    echo -n $(basename $path) >> durations-ljs
    echo -n " " >> durations-ljs
    soxi -D $path >> durations-ljs
done;

