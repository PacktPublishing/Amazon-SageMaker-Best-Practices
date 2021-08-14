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
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

class CsvDatasetSimple(Dataset):

    def __init__(self, csv_path, max_files=5):
        
        self.csv_path = csv_path
        if os.path.isfile(csv_path):
            self.count, self.tensors = self.get_line_data(csv_path)
            logger.debug(f"For {csv_path}, count = {self.count}")
        else:
            self.count, self.tensors = self.get_folder_line_data(csv_path, max_files)
            
    def get_folder_line_data(self, d, max_files):
        cnt = 0
        file_cnt = 0
        tensors = []
        for f in glob.glob(os.path.join(d, '*.csv')):
            fcnt, ftensors = self.get_line_data(f)
            cnt = cnt + fcnt
            tensors = tensors + ftensors
            file_cnt = file_cnt + 1
            if file_cnt > max_files:
                break
            
        return cnt, tensors
    
    def get_line_data(self, f):
        cnt = 0
        tensors = []
        with open(f) as F:
            f_lines = F.readlines()
            for l in f_lines:
                cnt = cnt + 1
                parts = l.split(',')
                tensors.append(tuple([torch.tensor( [float(f) for f in parts[1:]] ), torch.tensor(float(parts[0]))]))
        
        return cnt, tensors

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        logger.debug(f"Indices: {idx}")
        
        return self.tensors[idx]

    