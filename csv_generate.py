import sys
import os
from collections import defaultdict
import pandas as pd
import os
from multiprocessing import Pool

def load_enhancers():
    # read the excel file describing the enhancers
    enhancers = []
    labels = []
    df = pd.read_excel('Enhancer_Prediction/Tables/enhancers.xlsx', sheet_name='S1')
    for i in df.index:
        track = df['Chrom (mm10)'][i]
        start = int(df['Start (mm10)'][i])
        end = int(df['End (mm10)'][i])
        label = df['Limb-activity'][i]
        if label == 'negative':
            label = -1
        else:
            label = 1
        enhancers.append((track, start, end, label))
    return enhancers

enhancers = load_enhancers()
# output to a CSV
with open('enhancers.csv', 'w') as f:
    for enhancer in enhancers:
        f.write(','.join([str(x) for x in enhancer]) + '\n')
