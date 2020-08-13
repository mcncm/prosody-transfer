#!/bin/bash

for wav in LibriTTS/train-clean-360/*/*/*\.wav
do
    path=`echo -n $wav`
    echo -n $(basename $path) >> durations-libritts
    echo -n " " >> durations-libritts
    soxi -D $path >> durations-libritts
done;
