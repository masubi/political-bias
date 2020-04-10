#!/bin/bash

rm -rf data_combined
cp -r data data_combined
cp -r data_zombie/test/neg/ data_combined/test/neg/
cp -r data_zombie/dev/neg/ data_combined/dev/neg/
cp -r data_zombie/train/neg/ data_combined/train/neg/

ls -al data_combined/*/* | wc -l
