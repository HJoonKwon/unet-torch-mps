import os
import logging
import argparse
from tqdm import tqdm
from abc import ABC, abstractmethod
import torch
from torch.nn.utils import clip_grad_norm_


class Trainer(ABC):
    def __init__(self, config):
        self.num_epochs = config["num_epochs"]
        self.batch_size = config["batch_size"]
        self.lr = config["lr"]
        self.device = config["device"]

    @abstractmethod
    def train_iter(self):
        pass

    def train_epoch(self):
        pass

    @abstractmethod
    def evaluate_iter(self):
        pass

    def evaluate(self):
        pass

    def generate_dataloader(self, data_config):
        pass
