import sys
import os
from collections import defaultdict
import pandas as pd
import numpy as np

# class PiecewiseLookup:
#     def __init__(self):
#         self.table = defaultdict(list)
#
#     def hash(self, value):
#         return int(value / 1000)
#
#     def store(self, value):
#         h = self.hash(value)
#         self.table[]

# chromatin_signal = defaultdict(list)
#
# with open('test.bedGraph') as f:
#     for line in f:
#         tokens = line.split()
#         entry = (int(tokens[1]), int(tokens[2]), float(tokens[3]))
#         chromatin_signal[tokens[0]].append(entry)
#
# print("Done reading file, sorting...")

# for ch in chromatin_signal:
#     print(ch)
    # sorted_features = sorted(ch, key=lambda x:x[0])
    # print(sorted_features[:10])


def getBedGraphFiles():
    data_path = "bedGraph_data"
    results = []
    for filename in os.listdir(data_path):
        results.append(filename)

    return results

# now read the excel file
enhancers = []
df = pd.read_excel('Enhancer_Prediction/Tables/enhancers.xlsx', sheet_name='S1')
for i in df.index:
    track = df['Chrom (mm10)'][i]
    start = int(df['Start (mm10)'][i])
    end = int(df['End (mm10)'][i])
    enhancers.append((track, start, end))


# with open('test.bedGraph') as f:
#     for line in f:
#         tokens = line.split()
#         chrom = tokens[0]
#         start = int(tokens[1])
#         end = int(tokens[2])
#         value = float(tokens[3])
#         for idx, enhancer in enumerate(enhancers):
#             # match the track
#             if enhancer[0] == chrom:
#                 if enhancer[1] >= start and enhancer[1] < end:
#                     # we have found the start of an enhancer
#                     print("Track range: %d -> %d" % (start, end))
#                     print("Enhancer range: %d -> %d" % (enhancer[1], enhancer[2]))
#                     region_len = min(enhancer[2], end) - enhancer[1]
#                     enhancer_values[idx] += (value / region_len)
#                 elif enhancer[1] <= start and enhancer[2] >= end:
#                     # enhancer encapsulates this region
#                     print("Track range: %d -> %d" % (start, end))
#                     print("Enhancer range: %d -> %d" % (enhancer[1], enhancer[2]))
#                     region_len = end - start
#                     enhancer_values[idx] += (value / region_len)
#                 elif enhancer[1] <= start and enhancer[2] > start and enhancer[2] <= end:
#                     # enhancer ends in this segment, but started before it
#                     print("Track range: %d -> %d" % (start, end))
#                     print("Enhancer range: %d -> %d" % (enhancer[1], enhancer[2]))
#                     region_len = min(end, enhancer[2]) - start
#                     enhancer_values[idx] += (value / region_len)
#
# print(enhancer_values[:10])

bedGraphFiles = getBedGraphFiles()[1:]
print bedGraphFiles
enhancer_values = [0.0] * len(enhancers)
enhancer_matrix = np.zeros((len(bedGraphFiles), len(enhancers)))

print enhancer_matrix
