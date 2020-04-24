#!/bin/bash

for f in $(cat long-utterance-list)
do
    mv "$f" long-utterances
done;

