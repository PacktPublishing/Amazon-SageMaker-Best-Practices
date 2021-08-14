import os
from torch.utils.data import Dataset
import glob
import torch
import sys
import logging
import collections
import bisect
import time
import pandas as pd
from linecache import getline
from itertools import islice

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def _make_gen(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024*1024)
        
def rawgencount(filename):
    f = open(filename, 'rb')
    f_gen = _make_gen(f.raw.read)
    return sum( buf.count(b'\n') for buf in f_gen )


class CsvDatasetPd(Dataset):

    def __init__(self, csv_path, max_files=5):
        
        self.csv_path = csv_path
        if os.path.isfile(csv_path):
            self.folder = False
            self.count, fmap = self.get_line_count(csv_path)
            logger.debug(f"For {csv_path}, count = {self.count}")
        else:
            self.folder = True
            self.count, fmap = self.get_folder_line_count(csv_path, max_files)
            
        self.fmap = collections.OrderedDict(sorted(fmap.items()))
        self.max_files = max_files
            
    def get_folder_line_count(self, d, max_files):
        cnt = 0
        all_map = {}
        file_cnt = 0
        for f in glob.glob(os.path.join(d, '*.csv')):
            fcnt, _ = self.get_line_count(f)
            cnt = cnt + fcnt
            all_map[cnt] = f
            file_cnt = file_cnt + 1
            if file_cnt > max_files:
                break
            
        return cnt, all_map
    
    def get_line_count(self, f):
        count = rawgencount(f)
        return count, {count: f}

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        logger.debug(f"Indices: {idx}")
        
        # This gives us the index in the line counts greater than or equal to the desired index.
        # The map value for this line count is the file name containing that row.
        klist = list(self.fmap.keys())
        idx_m = bisect.bisect_left(klist, idx+1)
        
        # Grab the ending count of thisl file
        cur_idx = klist[idx_m]
        
        # grab the ending count of the previous file
        if idx_m > 0:
            prev_idx = klist[idx_m-1]
        else:
            prev_idx = 0
        
        # grab the file name for the desired row count
        fname = self.fmap[cur_idx]

        #with open(fname) as F:
        #    lines = list(islice(F, idx-prev_idx, idx-prev_idx+1))
        idx_line = getline(fname, idx - prev_idx +1)
        
        idx_parts = idx_line.split(',')

        return tuple([torch.tensor( [float(f) for f in idx_parts[1:]] ), torch.tensor(float(idx_parts[0]))])

    