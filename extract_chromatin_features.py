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

def process_file(bedGraphFile):
    enhancers = load_enhancers()
    enhancer_values = [0.0] * len(enhancers)
    enhancer_dict = {}
    for idx, enhancer in enumerate(enhancers):
        enhancer_dict[idx] = enhancer

    # print "This is the enhancer dict: ", enhancer_dict
    with open(bedGraphFile) as f:

        for line in f:
            tokens = line.split()
            chrom = tokens[0]
            start = int(tokens[1])
            end = int(tokens[2])
            value = float(tokens[3])
            to_remove = []

            print "NEXT ITERATION"
            print "--------------------------"
            # for idx, enhancer in enumerate(enhancers):
            for idx, enhancer in enhancer_dict.items():
                # match the track
                if enhancer[0] == chrom:

                    print "This is the start ENHANCER", enhancer[1]
                    print "BEDGRAPH START: ", start
                    print " ------------------------"
                    print "This is the end ENHANCER", enhancer[2]
                    print "BEDGRAPH END: ", end
                    print "-------------------------"

                    if enhancer[1] >= start and enhancer[1] < end:

                        print "Entered case 1"
                        # we have found the start of an enhancer
                        # print("Track range: %d -> %d" % (start, end))
                        # print("Enhancer range: %d -> %d" % (enhancer[1], enhancer[2]))
                        region_len = min(enhancer[2], end) - enhancer[1]
                        enhancer_values[idx] += (value / region_len)
                        if (enhancer[2] <= end + 1):
                            # we are done processing this one
                            to_remove.append(idx)
                    elif enhancer[1] <= start and enhancer[2] >= end:

                        print "Entered case 2"
                        # enhancer encapsulates this whole region
                        # print("Track range: %d -> %d" % (start, end))
                        # print("Enhancer range: %d -> %d" % (enhancer[1], enhancer[2]))
                        region_len = end - start
                        enhancer_values[idx] += (value / region_len)
                        if (enhancer[2] <= end + 1):
                            # we are done processing this one
                            to_remove.append(idx)
                    elif enhancer[1] <= start and enhancer[2] > start and enhancer[2] <= end:

                        print "Entered case 3"
                        # enhancer ends in this segment, but started before it
                        # print("Track range: %d -> %d" % (start, end))
                        # print("Enhancer range: %d -> %d" % (enhancer[1], enhancer[2]))
                        region_len = min(end, enhancer[2]) - start
                        enhancer_values[idx] += (value / region_len)
                        if (enhancer[2] <= end + 1):
                            # we are done processing this one
                            to_remove.append(idx)

            for enhancer_idx in to_remove:
                del enhancer_dict[enhancer_idx]

    out_filename = ''.join(bedGraphFile.split('.')[:-1])
    np.save(out_filename, np.array(enhancer_values))
    return enhancer_values

def process_all_files(directory):
    enhancers = load_enhancers()
    files = os.listdir(directory)
    # process_bed_file = lambda x: process_file(x, enhancers)
    bed_files = [directory + '/' + x for x in files if x[0] != "."]
    print bed_files
    for f in bed_files:
        values = process_file(f)
        print values
        break
        # process_bed_file(f)
    # with Pool(3) as p:
    #     p.map(process_file, bed_files)
    print("DONE!")

process_all_files('bedGraph_data')
