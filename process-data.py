import os
import subprocess

runFile = "./bigWigToBedGraph"
dataDir = "/Users/samirsen/Documents/Samir/Junior_year/enhancer-prediction/data_bw"
outDir = "/Users/samirsen/Documents/Samir/Junior_year/enhancer-prediction/bedGraph_data"

identifier_h3k27 = "H3K27ac"
identifier_p300 = "p300"
identifier_DNASE = "Dnase"

for i, filename in enumerate(os.listdir(dataDir)):
    print filename

    if "bed" in filename:
        continue

    outFile = identifier_h3k27

    if identifier_DNASE in filename:
        outFile = identifier_DNASE

    elif identifier_p300 in filename:
        outFile = identifier_p300

    outputFile = outDir + "/" + outFile + "_" + str(i)

    args = runFile + dataDir + "/" + filename + " " + outputFile

    print args

    subprocess.call('bigWigToBedGraph', cwd="/Users/samirsen/Documents/Samir/Junior_year/enhancer-prediction")
