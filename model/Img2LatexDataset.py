import os
import torch
import numpy as np
from torch.utils.data import Dataset, Dataloader


class Img2LatexDataset(Dataset):

    def __init__(self, formulas_file, root_dir):
        """
        Args:
            formulas_file (String): Path to the formulas file
            root_dir (String): Directory with all the images in pdf format
        """

        pass
        