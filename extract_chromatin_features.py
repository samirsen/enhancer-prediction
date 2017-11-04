import sys
import os
from collections import defaultdict
import pandas as pd
import numpy as np

# read the excel file describing the enhancers
def getEnhancers():
    enhancers = []
    df = pd.read_excel('Enhancer_Prediction/Tables/enhancers.xlsx', sheet_name='S1')
    for i in df.index:
        track = df['Chrom (mm10)'][i]
        start = int(df['Start (mm10)'][i])
        end = int(df['End (mm10)'][i])
        enhancers.append((track, start, end))

    return enhancers

def computeIntegration(filename, enhancers):
    enhancer_values = [0.0] * len(enhancers)

    with open('test.bedGraph') as f:
        for line in f:
            tokens = line.split()
            chrom = tokens[0]
            start = int(tokens[1])
            end = int(tokens[2])
            value = float(tokens[3])
            for idx, enhancer in enumerate(enhancers):
                # match the track
                if enhancer[0] == chrom:
                    if enhancer[1] >= start and enhancer[1] < end:
                        # we have found the start of an enhancer
                        print("Track range: %d -> %d" % (start, end))
                        print("Enhancer range: %d -> %d" % (enhancer[1], enhancer[2]))
                        region_len = min(enhancer[2], end) - enhancer[1]
                        enhancer_values[idx] += (value / region_len)
                    elif enhancer[1] <= start and enhancer[2] >= end:
                        # enhancer encapsulates this region
                        print("Track range: %d -> %d" % (start, end))
                        print("Enhancer range: %d -> %d" % (enhancer[1], enhancer[2]))
                        region_len = end - start
                        enhancer_values[idx] += (value / region_len)
                    elif enhancer[1] <= start and enhancer[2] > start and enhancer[2] <= end:
                        # enhancer ends in this segment, but started before it
                        print("Track range: %d -> %d" % (start, end))
                        print("Enhancer range: %d -> %d" % (enhancer[1], enhancer[2]))
                        region_len = min(end, enhancer[2]) - start
                        enhancer_values[idx] += (value / region_len)

    return enhancer_values


bedGraphFiles = getBedGraphFiles()[1:]
enhancers = getEnhancers()
enhancer_values = [0.0] * len(enhancers)
#enhancer_matrix = np.zeros((len(bedGraphFiles), len(enhancers)))
enhancer_matrix = np.array()

for bedfile in bedGraphFiles:
    enhancer_values = computeIntegration(bedfile, enhancers)
    enhancer_matrix.vstack(enhancer_values)
