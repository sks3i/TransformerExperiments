"""
Basic single host trainer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import subprocess
import threading

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
        self.to_device()

    def to_device(self):
        self.model.to(self.device)
        self.criterion.to(self.device)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        for epoch in range(num_epochs):
            self.model.train()
            for i, batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                out = self.model(batch)
                loss = self.criterion(out, batch['tgt']['input_ids'])
                loss.backward()
                self.optimizer.step()                
                self.writer.add_scalar("loss", loss.item(), epoch * len(train_loader) + i)
                print(f"epoch: {epoch}, {i}/{len(train_loader)} loss: {loss.item()}")
            
            self.evaluate(val_loader, epoch)

    def evaluate(self, val_loader: DataLoader, epoch: int):
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                loss = self.criterion(self.model(batch), batch['tgt']['input_ids'])
                self.writer.add_scalar("loss", loss.item(), epoch * len(val_loader) + i)
                print(f"epoch: {epoch}, {i}/{len(val_loader)} loss: {loss.item()}")
