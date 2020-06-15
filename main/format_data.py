"""
1, This writes the file-lookup .csv file `./data_files.csv` from a path to 
the folders containing the recordings. All Subsequent scripts use this 
table to lookup file locations. 
"""
import pandas as pd
from glob import glob
from os import listdir
import os
from config import path_to_data

# load data (change if "data" folder is not in the same folder as this file)
#path_to_data = '../data/'

days = [14, 15, 16, 18, 19]
path_to_data = [path_to_data+'/2018-08-{}'.format(d) for d in days]
# list of all files
files = sorted([f for path in path_to_data for f in glob(path  + '*.*')])

# slice out the recording id from the file to construct a dataframe
recording_ids = []
for i, f in enumerate(files):
    undersc1 = f.rfind('_')
    undersc2 = f[:undersc1].rfind('_') +1
    rec_id = f[undersc2: undersc1]
    if rec_id not in recording_ids:
        recording_ids.append(rec_id)

# make a dataframe that has the recording id as rows, and the filetypes as columns
# each entry is the path to the respective file so you can easily get the files you need.
filetypes = ['DAQmx', 'SdrCarrierFreq', 'SdrChannelList', 'SdrChannels',
             'SdrReceiveFreq', 'SdrSignalStrength','log']
data_files = pd.DataFrame(index=pd.Index(recording_ids, name='rec_id'), columns=filetypes)

# populate dataframe
for filetype in data_files.columns:
    #sub_files = [f for f in files if filetype in f]
    #sub_files = [[path+'/'+f] for path in path_to_data for f in listdir(path) if filetype in f]
    sub_files = [os.path.join(path,f) for path in path_to_data for f in listdir(path) if filetype in f]     # modivication josua
    data_files[filetype] = sub_files
print(data_files)
# write a csv table to easily index the files you need
# load this table with pd.read_csv('./data_files.csv', index_col='rec_id')
data_files.to_csv('../data_files.csv')