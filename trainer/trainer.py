"""
Basic single host trainer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(
            self,
            model: nn.Module,
            optimizer: optim.Optimizer,
            criterion: nn.Module,
            device: torch.device,
            log_dir: str,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        for epoch in range(num_epochs):
            self.model.train()
            for i, batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(batch), batch)
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar("loss", loss.item(), epoch * len(train_loader) + i)
            
            self.evaluate(val_loader, epoch)

    def evaluate(self, val_loader: DataLoader, epoch: int):
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                loss = self.criterion(self.model(batch), batch)
                self.writer.add_scalar("loss", loss.item(), epoch * len(val_loader) + i)

    
