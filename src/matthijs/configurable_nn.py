import torch
import torch.nn as nn


from dataclasses import dataclass
@dataclass
class NNLayerConfig:
    """Class for storing config and results for a layer."""
    accuracy_train: float
    accuracy_test: float
    num_params: int
    c1_ksize: int
    c1_stride: int
    c2_ksize: int
    c2_stride: int
    num_filters: int


class ConfigurableNN (nn.Module):

    def __init__(self, config:NNLayerConfig, example_input):
        super().__init__()

        self.config = config
        self.maxpool_ksize = 2 # Hardcoded for now.
        # print(self.config)

        # Convolutions are set separately
        self.convolutions = nn.Sequential(
            nn.Conv2d(  1, # First one is always 1
                        self.config.num_filters, 
                        kernel_size = self.config.c1_ksize,
                        stride = self.config.c1_stride,
                        padding = 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.maxpool_ksize),
            nn.Conv2d(  self.config.num_filters,
                        self.config.num_filters,
                        kernel_size = self.config.c2_ksize,
                        stride = self.config.c2_stride,
                        padding = 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.maxpool_ksize)

        )

        flat_size = self.calculate_flat_size(example_input)
        
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def calculate_flat_size(self, example_input):
        current = example_input
        for conv in self.convolutions:
            current = conv(current)
        flatten = nn.Flatten()
        return flatten(current).shape[1]

    # Forward is default for now.
    def forward(self, x):
        x = self.convolutions(x)
        logits = self.dense(x)
        return logits