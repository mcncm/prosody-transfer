#!/bin/bash

for tsv in LibriTTS/train-clean-360/*/*/*.trans\.tsv
do
    cat $tsv >> prepped
done;

