# Standalone version of the experiment in matthijs.ipynb
# as running it in Jupyter in VSCode causes VSCode to lock up.

import torch
import torch.nn as nn
import seaborn as sns
import sys
import gin
import itertools
from pathlib import Path
sys.path.insert(0, "./")
print(sys.path)
from src.data import make_dataset
from src.models import imagemodels
from src.models import train_model

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Load config for this notebook
gin.add_config_file_search_path("./notebooks")
gin.parse_config_file("./notebooks/cnn.gin")

# Get MNIST data
train_dataloader, test_dataloader = make_dataset.get_MNIST()

# Set some other parameters
import torch.optim as optim
from src.models import metrics
optimizer = optim.Adam
loss_fn = torch.nn.CrossEntropyLoss()
accuracy = metrics.Accuracy()
from src.models import train_model

from src.matthijs.configurable_nn import ConfigurableNN, NNLayerConfig
from src.matthijs.dump_nn_model import dump_nn_model_steps

configs = []  # List of NNLayerConfig objects

# Configuration for the experiment. We ignore padding for now.
# These settings take some tweaking to avoid ending up with runtime
# errors when the sizes end up too small.
kernel_sizes = [1, 2, 3]
filter_nums = [16, 32, 64]
strides = [1, 2]

for config in itertools.product(kernel_sizes, # kernel size in convolution 1
                                strides, # stride size in convolution 1
                                kernel_sizes, # kernel size in convolution 2
                                strides, # stride size in convolution 2
                                filter_nums, # number of filters to use
                                ):
    configs.append(NNLayerConfig(
        accuracy_train = 0,
        accuracy_test = 0,
        num_params = 0,
        c1_ksize = config[0],
        c1_stride = config[1],
        c2_ksize = config[2],
        c2_stride = config[3],
        num_filters = config[4],
    ))

# Now run the experiment
num_epochs = 2 # During development keep very low. 10 is for real tests
num_epochs = 10
learning_rate = 1e-3

xtrain, ytrain = next(iter(train_dataloader))
xtest, ytest = next(iter(train_dataloader))

for config in configs:
    model = ConfigurableNN(config, xtrain).to(device)
    # dump_nn_model_steps(model, x)
    # continue;
    model = train_model.trainloop(
        epochs=num_epochs,
        model=model,
        optimizer=optimizer,
        learning_rate=learning_rate,
        loss_fn=loss_fn,
        metrics=[accuracy],
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        log_dir="../models/test/",
        train_steps=len(train_dataloader),
        eval_steps=len(test_dataloader),
    )
    config.num_params = train_model.count_parameters(model)
    yhat_train = model(xtrain)
    config.accuracy_train = accuracy(ytrain, yhat_train).item()
    yhat_test = model(xtest)
    config.accuracy_test = accuracy(ytest, yhat_test).item()
    # break # During development

# Once we're done training, store the config data in a csv file.
import pandas as pd
# import matplotlib as plt

df = pd.DataFrame(configs)
df['config_num'] = df.index
df['config_label'] = "Config: " + df["c1_ksize"].astype(str) + "/" + \
    df["c1_stride"].astype(str) + ", " + \
    df["c2_ksize"].astype(str) + "/" + \
    df["c2_stride"].astype(str) + " flt #" + \
    df["num_filters"].astype(str)
df.to_csv("matthijs.csv", index = False)
