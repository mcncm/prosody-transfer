#!/bin/bash

for f in $(cat short-utterance-list)
do
    mv "$f" short-utterances
done;

