import argparse
import numpy as np
import os
import sys
import logging
import json
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from model_pytorch import TabularNet
from csv_loader_simple import CsvDatasetSimple

import smdistributed.dataparallel.torch.distributed as dist
from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

class CUDANotFoundException(Exception):
    pass


dist.init_process_group()


def parse_args():
    """
    Parse arguments passed from the SageMaker API
    to the container
    """

    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.01)

    # Data directories
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))

    # Model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    
    parser.add_argument(
        "--save-model", action="store_true", default=False, help="For Saving the current Model"
    )

    return parser.parse_known_args()

def train(args):
    """
    Train the PyTorch model
    """

    cat_mask=[False,True,True,True,True,False,True,True,True,True,True,False,False,False,False,False,False,False]
    train_ds = CsvDatasetSimple(args.train)
    test_ds = CsvDatasetSimple(args.test)

    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    logger.info(
        "batch_size = {}, epochs = {}, learning rate = {}".format(batch_size, epochs, learning_rate)
    )
    logger.info(f"World size: {args.world_size}")
    logger.info(f"Rank: {args.rank}")
    logger.info(f"Local Rank: {args.local_rank}")
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds, num_replicas=args.world_size, rank=args.rank
    )
    logger.debug(f"Created distributed sampler")

    train_dl = DataLoader(train_ds, batch_size, shuffle=False, 
                          num_workers=0,
                          pin_memory=True,
                          drop_last=True, sampler=train_sampler)
    logger.debug(f"Created train data loader")

    model = TabularNet(n_cont=9, n_cat=9, 
                          cat_mask = cat_mask, 
                          cat_dim=[0,2050,13,5,366,0,50000,50000,50000,50000,50,0,0,0,0,0,0,0], 
                          y_min = 0., y_max = 1.)
    logger.debug("Created model")
    logger.debug(model)
    model = DDP(model.to(device), broadcast_buffers=False)
    logger.debug("created DDP")
    torch.cuda.set_device(args.local_rank)
    logger.debug("Set device on CUDA")
    model.cuda(args.local_rank)
    logger.debug(f"Set model CUDA to {args.local_rank}")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    logger.debug("Created loss fn and optimizer")

    model.train()
    for epoch in range(epochs):
        batch_no = 0
        for x_train_batch, y_train_batch in train_dl:
            logger.debug(f"Working on batch {batch_no}")
            x_train_batch_d = x_train_batch.to(device)
            y_train_batch_d = y_train_batch.to(device)
            
            optimizer.zero_grad()
            logger.debug("Did optimizer grad")
            
            logger.debug(f"Training on shape {x_train_batch.shape}")
            y = model(x_train_batch_d.float())
            logger.debug("Did forward pass")
            loss = criterion(y.flatten(), y_train_batch_d.float())
            logger.debug("Got loss")
            if batch_no % 50 == 0:
                logger.info(f"batch {batch_no} -> loss: {loss}")
            
            loss.backward()
            logger.debug("Did backward step")
            optimizer.step()
            logger.debug(f"Did optimizer step, batch {batch_no}")
            batch_no +=1
        epoch += 1
        logger.info(f"epoch: {epoch} -> loss: {loss}")

    # evalutate on test set
    if args.rank == 0:
        logger.info(f"Starting test eval on rank 0")
        model.eval()
        test_dl = DataLoader(test_ds, batch_size, drop_last=True, shuffle=False)
        logger.info(f"Loaded test data set")
        with torch.no_grad():
            mse = 0.
            batch_no = 0
            for x_test_batch, y_test_batch in test_dl:
                x_test_batch_d = x_test_batch.to(device)
                y_test_batch_d = y_test_batch.to(device)
                
                y = model(x_test_batch_d.float())
                mse += F.mse_loss(y, y_test_batch_d, reduction="sum")
                #mse = mse + ((y - y_test_batch_d) ** 2).sum() / x_test_batch.shape[0]
                
                if batch_no % 50 == 0:
                    logger.info(f"batch {batch_no} -> MSE: {mse}")
                batch_no +=1
                
        mse = mse / len(test_dl.dataset)
        logger.info(f"Test MSE: {mse}")

    if args.rank == 0:
        logger.info("Saving model on rank 0")
        torch.save(model.state_dict(), args.model_dir + "/model.pth")


if __name__ == "__main__":

    args, _ = parse_args()
    args.world_size = dist.get_world_size()
    args.rank = rank = dist.get_rank()
    args.local_rank = local_rank = dist.get_local_rank()
    args.batch_size //= args.world_size // 8
    args.batch_size = max(args.batch_size, 1)
    
    if not torch.cuda.is_available():
        raise CUDANotFoundException(
            "Must run smdistributed.dataparallel MNIST example on CUDA-capable devices."
        )
        
    torch.manual_seed(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(args)
