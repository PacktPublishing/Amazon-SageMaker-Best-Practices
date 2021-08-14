import os
from torch.utils.data import Dataset
import glob
import torch
import sys
import logging
import collections
import bisect

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

class CsvDataset(Dataset):

    def __init__(self, csv_path):
        
        self.csv_path = csv_path
        if os.path.isfile(csv_path):
            self.folder = False
            self.count, fmap, self.line_offset = self.get_line_count(csv_path)
            logger.debug(f"For {csv_path}, count = {self.count}")
        else:
            self.folder = True
            self.count, fmap, self.line_offset = self.get_folder_line_count(csv_path)
            
        self.fmap = collections.OrderedDict(sorted(fmap.items()))
            
    def get_folder_line_count(self, d):
        cnt = 0
        all_map = {}
        all_lc = {}
        for f in glob.glob(os.path.join(d, '*.csv')):
            fcnt, _, line_offset = self.get_line_count(f)
            cnt = cnt + fcnt
            all_map[cnt] = f
            all_lc.update(line_offset)
        return cnt, all_map, all_lc
    
    def get_line_count(self, f):
        with open(f) as F:
            line_offset = []
            offset = 0
            count = 0
            for line in F:
                line_offset.append(offset)
                offset += len(line)
                count = count + 1
        
        return count, {count: f}, {f: line_offset}

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

        loff = self.line_offset[fname]
        with open(fname) as F:
            F.seek(loff[idx - prev_idx])
            idx_line = F.readline()    
        
        idx_parts = idx_line.split(',')

        return tuple([torch.tensor( [float(f) for f in idx_parts[1:]] ), torch.tensor(float(idx_parts[0]))])