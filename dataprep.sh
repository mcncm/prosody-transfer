#!/bin/bash

for file in blizzard-txt/*
do
    echo -n $file >> prepped
    echo -en "|" >> prepped
    cat $file >> prepped
    echo "" >> prepped
done;

