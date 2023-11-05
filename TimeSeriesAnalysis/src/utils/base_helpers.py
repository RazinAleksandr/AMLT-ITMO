import numpy as np
import torch
import random


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# Function to parse the data and get the second value
def parse_data(file_content):
    # Split the content by lines
    lines = file_content.split('\n')
    
    # Extract the second value from each line
    second_values = [line.split()[1] for line in lines if line]
    
    return second_values