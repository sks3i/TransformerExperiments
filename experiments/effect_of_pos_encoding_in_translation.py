"""
This experiment is to test the effect of positional encoding in translation.
"""

import os
import sys
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data.dataset import get_data_loaders
from data.dataset import TranslationDataCollator
from model.transformer import TransformerEncoder, TransformerDecoder
from trainer.trainer import Trainer
from transformers import PreTrainedTokenizerFast


class ExperimentConfig:
    def __init__(self, experiment_name: str):
        # Model architecture
        self.d_model = 512
        self.n_heads = 8 
        self.d_ff = 2048
        self.n_layers = 6
        self.dropout = 0.1
        self.norm_order = "post"

        # Training
        self.batch_size = 136
        self.num_epochs = 2
        self.learning_rate = 0.0001
        self.warmup_steps = 4000

        # Data
        self.max_seq_length = 100
        self.vocab_size = 32000
        self.tokenizer_file = "artifacts/tokenizers/en_de_tokenizer.json"

        # Positional encoding
        self.use_positional_encoding = True
        self.max_len = 5000

        # Logging
        self.experiment_name = experiment_name
        self.log_dir = f"logs/translation_pos_encoding/{self.experiment_name}"
        self.save_dir = "checkpoints/translation_pos_encoding"
        self.log_interval = 100

    def get_model_params(self):
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "max_len": self.max_len,
            "vocab_size": self.vocab_size,
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

    def forward(self, logits, target):
        """
        logits: [batch, seq_len, vocab_size]
        target: [batch, seq_len]
        """
        return self.criterion(logits.view(-1, logits.size(-1)), target.view(-1))


class Model(nn.Module):
    def __init__(self, config: ExperimentConfig):
        super(Model, self).__init__()
        self.config = config
        self.device = get_device()

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = TransformerEncoder(**config.get_model_params())
        self.decoder = TransformerDecoder(**config.get_model_params())
        
        self.linear = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, batch):
        # Extract source and target from the batch
        src = batch['src']['input_ids']
        tgt = batch['tgt']['input_ids']
        
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        
        if self.config.use_positional_encoding:
            src = src + get_positional_encoding(self.config.d_model, src.size(1))
            tgt = tgt + get_positional_encoding(self.config.d_model, tgt.size(1))
        
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        output = self.linear(output)
        return output
    
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_tokenizer(config: ExperimentConfig):
    special_tokens = {
        "pad_token": "[PAD]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "mask_token": "[MASK]"
    }
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=config.tokenizer_file,
        model_max_length=config.max_seq_length,
    )
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer


def base_experiment(config: ExperimentConfig):
    device = get_device()
    tokenizer = get_tokenizer(config)
    collate_fn = TranslationDataCollator(
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        device=device
    )
    data_loaders = get_data_loaders(
        "translation",
        config.batch_size,
        collate_fn
    )
    model = Model(config)
    criterion = Criterion()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    trainer = Trainer(model, optimizer, criterion, device, config.log_dir)
    return trainer, data_loaders


def positional_encoding_experiment():
    config = ExperimentConfig(experiment_name="positional_encoding")
    trainer, data_loaders = base_experiment(config)
    trainer.train(data_loaders["train"], data_loaders["val"], config.num_epochs)


def no_positional_encoding_experiment():
    config = ExperimentConfig(experiment_name="no_positional_encoding")
    config.use_positional_encoding = False
    trainer, data_loaders = base_experiment(config)
    trainer.train(data_loaders["train"], data_loaders["val"], config.num_epochs)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--use_positional_encoding", action="store_true")
    args = args.parse_args()

    if args.use_positional_encoding:
        print("Using positional encoding")
        positional_encoding_experiment()
    else:
        print("Not using positional encoding")
        no_positional_encoding_experiment()
