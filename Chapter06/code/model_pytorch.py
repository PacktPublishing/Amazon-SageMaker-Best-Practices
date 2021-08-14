import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

class TabularNet(nn.Module):

    def __init__(self, n_cont, n_cat, emb_sz = 100, dropout_p = 0.1, layers=[200,100], cat_mask=[], cat_dim=[], y_min = 0., y_max = 1.):
        
        super(TabularNet, self).__init__()
        
        self.cat_mask = cat_mask
        self.cat_dim = cat_dim
        self.y_min = y_min
        self.y_max = y_max
        self.n_cat = n_cat
        self.n_cont = n_cont
        
        emb_dim = []
        for ii in range(len(cat_mask)):
            if cat_mask[ii]:
                c_dim = cat_dim[ii]
                emb_dim.append(c_dim)
                #emb = nn.Embedding(c_dim, emb_sz)
                #self.embeddings.append(emb)
                #setattr(self, 'emb_{}'.format(ii), emb)
                
        self.embeddings = nn.ModuleList([nn.Embedding(c_dim, emb_sz) for c_dim in emb_dim])
                
        modules = []
        prev_size = n_cont + n_cat * emb_sz
        for l in layers:
            modules.append(nn.Linear(prev_size, l))
            modules.append(nn.BatchNorm1d(l))
            modules.append(nn.Dropout(dropout_p))
            modules.append(nn.ReLU(inplace=True))
            prev_size = l
        modules.append(nn.BatchNorm1d(prev_size))
        modules.append(nn.Dropout(dropout_p))
        modules.append(nn.Linear(prev_size, 1))
        modules.append(nn.Sigmoid())
        
        self.m_seq = nn.Sequential(*modules)
        self.emb_drop = nn.Dropout(dropout_p)
        self.bn_cont = nn.BatchNorm1d(n_cont)

    def forward(self, x_in):
        
        logger.debug(f"Forward pass on {x_in.shape}")
        x = None
        ee = 0
        for ii in range(len(self.cat_mask)):

            if self.cat_mask[ii]:
                logger.debug(f"Embedding: {self.embeddings[ee]}, input: {x_in[:,ii]}")
                logger.debug(f"cat Device for x_in: {x_in.get_device()}")
                logger.debug(f"cat Device for x_in slice: {x_in[:,ii].get_device()}")
                logger.debug(f"cat Device for embed: {next(self.embeddings[ee].parameters()).get_device()}")
                x_e = self.embeddings[ee](x_in[:,ii].to(device = x_in.get_device(), dtype= torch.long))
                logger.debug(f"cat Device for x_e: {x_e.get_device()}")
                logger.debug(f"cat x_e = {x_e.shape}")
                if x is None:
                    x = x_e
                else:
                    x = torch.cat([x, x_e], 1)
                logger.debug(f"cat Device for x: {x.get_device()}")
                x = self.emb_drop(x)
                logger.debug(f"cat Device for x: {x.get_device()}")
                logger.debug(f"cat x = {x.shape}")
                ee = ee + 1
            else:
                logger.debug(f"cont Device for x_in: {x_in.get_device()}")
                x_cont = x_in[:, ii] # self.bn_cont(x_in[:, ii])
                logger.debug(f"cont Device for x_cont: {x_cont.get_device()}")
                logger.debug(f"cont x_cont = {x_cont.shape}")
                if x is None:
                    x = torch.unsqueeze(x_cont, 1)
                else:
                    x = torch.cat([x, torch.unsqueeze(x_cont, 1)], 1)
                logger.debug(f"cont Device for x: {x.get_device()}")
                logger.debug(f"cont x = {x.shape}")
                 
        return self.m_seq(x) * (self.y_max - self.y_min) + self.y_min