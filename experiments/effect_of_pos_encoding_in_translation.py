"""
This experiment is to test the effect of positional encoding in translation.
"""

import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data.dataset import get_data_loaders
from model.transformer import TransformerEncoder, TransformerDecoder
from trainer.trainer import Trainer


class ExperimentConfig:
    def __init__(self):
        # Model architecture
        self.d_model = 512
        self.n_heads = 8 
        self.d_ff = 2048
        self.n_layers = 6
        self.dropout = 0.1
        self.norm_order = "post"

        # Training
        self.batch_size = 32
        self.num_epochs = 100
        self.learning_rate = 0.0001
        self.warmup_steps = 4000

        # Data
        self.max_seq_length = 100
        self.vocab_size = 32000

        # Positional encoding
        self.use_positional_encoding = True
        self.max_len = 5000

        # Logging
        self.log_dir = "logs/translation_pos_encoding"
        self.save_dir = "checkpoints/translation_pos_encoding"
        self.log_interval = 100

    def get_model_params(self):
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff, 
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "norm_order": self.norm_order
        }

    def get_training_params(self):
        return {
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps
        }


def get_positional_encoding(d_model, max_len):
    """
    Get positional encoding for the input sequence. 
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    return pe


class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.criterion(output, target)


class Model(nn.Module):
    def __init__(self, config: ExperimentConfig):
        super(Model, self).__init__()
        self.config = config
        self.encoder = TransformerEncoder(**config.get_model_params())
        self.decoder = TransformerDecoder(**config.get_model_params())
        
        self.linear = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, src, tgt):
        if self.config.use_positional_encoding:
            src = src + get_positional_encoding(self.config.d_model, src.size(1))
            tgt = tgt + get_positional_encoding(self.config.d_model, tgt.size(1))
        
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        output = self.linear(output)
        return output
    
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def base_experiment(config: ExperimentConfig):
    device = get_device()
    data_loaders = get_data_loaders("translation", config.batch_size)
    model = Model(config)
    criterion = Criterion()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    trainer = Trainer(model, optimizer, criterion, device, config.log_dir)
    return trainer, data_loaders


def positional_encoding_experiment():
    config = ExperimentConfig()
    trainer, data_loaders = base_experiment(config)
    trainer.train(data_loaders["train"], data_loaders["val"], config.num_epochs)


def no_positional_encoding_experiment():
    config = ExperimentConfig()
    config.use_positional_encoding = False
    trainer, data_loaders = base_experiment(config)
    trainer.train(data_loaders["train"], data_loaders["val"], config.num_epochs)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--use_positional_encoding", action="store_true")
    args = args.parse_args()

    if args.use_positional_encoding:
        positional_encoding_experiment()
    else:
        no_positional_encoding_experiment()