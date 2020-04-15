#!/bin/bash

rm -rf ../../data/raw/data_combined
cp -r ../../data/raw/data ../../data/raw/data_combined
cp -r ../../data/raw/data_zombie/test/neg/ ../../data/raw/data_combined/test/neg/
cp -r ../../data/raw/data_zombie/dev/neg/ ../../data/raw/data_combined/dev/neg/
cp -r ../../data/raw/data_zombie/train/neg/ ../../data/raw/data_combined/train/neg/

ls -al ../../data/raw/data_combined/*/* | wc -l
