import subprocess
import os

def get_all_files(directory):
    files = os.listdir(directory)
    bw_files = [x for x in files if x.endswith('.bw')]
    return bw_files

# subprocess.call('./bw2bg', shell=True)
def process_file(in_dir, out_dir, filename):
    outname = out_dir + '/' + ''.join(filename.split('.')[:-1]) + '.bedGraph'
    subprocess.call('./bw2bg ' + in_dir + '/' + filename + ' ' + outname, shell=True)

data_dir = 'raw_data'
out_dir = 'bed_data'
files = get_all_files(data_dir)
for f in files:
    process_file(data_dir, out_dir, f)
