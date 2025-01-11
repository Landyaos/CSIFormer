import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import hdf5storage
import matplotlib.pyplot as plt
import torch.optim as optim



def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path)
    model.to(device)
    model.eval()
    return model

def infer(model, input_data):
    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        output = model(input_tensor)
    return output
