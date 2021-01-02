import os
import torch
from torch.utils.data import Dataset
from skimage import io


class DatasetGenerator(Dataset):

    def __init__(self, formulas_file, root_dir, transform=None):
        """
        Args:
            formulas_file (String): Path to the formulas file
            root_dir (String): Directory with all the images in png format
        """
        self.formulas = open(formulas_file, 'r').read().split('\n')[:-1]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.formulas)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, "{}.png".format(idx))

        image = io.imread(img_name)
        formula = self.formulas[idx]

        sample = {'image': image, 'formula': formula}

        if self.transform:
            sample = self.transform(sample)

        return sample
