#!/bin/bash

for file in /Users/samirsen/Documents/Samir/Junior_year/enhancer-prediction/data_bw/*;do
  echo "Processing $filename"

  ouputFile = ${file##*/} + "_processed"
  ./Users/samirsen/Documents/Samir/Junior_year/enhancer-prediction/bigWigToBedGraph $filename $outputFile

  echo "Done creating - $outputFile" 
done
