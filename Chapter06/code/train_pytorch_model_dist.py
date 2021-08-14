import argparse
import numpy as np
import os
import sys
import logging
import json
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchnet.dataset import SplitDataset
from torchvision import datasets

from torch.utils.data import DataLoader, TensorDataset
from model_pytorch import TabularNet
from csv_loader_simple import CsvDatasetSimple

# SMP: Import and initialize SMP API.
import smdistributed.modelparallel.torch as smp


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

smp.init()


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

    return parser.parse_known_args()


# smdistributed: Define smp.step. Return any tensors needed outside.
@smp.step
def train_step(model, data, target):
    #logger.info("**** TRAIN_STEP method target is {} ".format(target))
    #print("target is : ", target)
    output = model(data)
    long_target = target.long()
    #loss = F.nll_loss(output, target, reduction="mean")
    loss = F.nll_loss(output, long_target, reduction="mean")
    model.backward(loss)
    return output, loss

def train(model, device, train_loader, optimizer):
    print("Into init_train")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #logger.info("**** TRAIN method target is {}".format(target))
        # smdistributed: Move input tensors to the GPU ID used by the current process,
        # based on the set_device call.
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # Return value, loss_mb is a StepOutput object
        _, loss_mb = train_step(model, data, target)

        # smdistributed: Average the loss across microbatches.
        loss = loss_mb.reduce_mean()

        optimizer.step()
     
    
def init_train():
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
    
    # smdistributed: initialize the backend
    smp.init()
    
    # smdistributed: Set the device to the GPU ID used by the current process.
    # Input tensors should be transferred to this device.
    torch.cuda.set_device(smp.local_rank())
    device = torch.device("cuda")
    
    # smdistributed: Download only on a single process per instance.
    # When this is not present, the file is corrupted by multiple processes trying
    # to download and extract at the same time
    #dataset = datasets.MNIST("../data", train=True, download=False)
    dataset = train_ds

    # smdistributed: Shard the dataset based on data-parallel ranks
    if smp.dp_size() > 1:
        partitions_dict = {f"{i}": 1 / smp.dp_size() for i in range(smp.dp_size())}
        dataset = SplitDataset(dataset, partitions=partitions_dict)
        dataset.select(f"{smp.dp_rank()}")
    
    
    # smdistributed: Set drop_last=True to ensure that batch size is always divisible
    # by the number of microbatches
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, drop_last=True)

    model = TabularNet(n_cont=9, n_cat=9, 
                          cat_mask = cat_mask, 
                          cat_dim=[0,2050,13,5,366,0,50000,50000,50000,50000,50,0,0,0,0,0,0,0], 
                          y_min = 0., y_max = 1.)
    
    logger.debug(model)
    
    optimizer = optim.Adadelta(model.parameters(), lr=4.0)
    
    
    # SMP: Instantiate DistributedModel object using the model.
    # This handles distributing the model among multiple ranks
    # behind the scenes
    # If horovod is enabled this will do an overlapping_all_reduce by
    # default.
    
    # smdistributed: Use the DistributedModel container to provide the model
    # to be partitioned across different ranks. For the rest of the script,
    # the returned DistributedModel object should be used in place of
    # the model provided for DistributedModel class instantiation.
    model = smp.DistributedModel(model)
    
    optimizer = smp.DistributedOptimizer(optimizer)
    
    train(model, device, train_loader, optimizer)

    torch.save(model.state_dict(), args.model_dir + "/model.pth")
    

if __name__ == "__main__":

    args, _ = parse_args()
        
    init_train()
