import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.path_A = os.path.join(self.root, 'A')
        self.dataset_A_filenames = sorted(make_dataset(self.path_A))
        self.path_B = os.path.join(self.root, 'B')
        self.dataset_B_filenames = sorted(make_dataset(self.path_B))

        # assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):

        filename_A = self.dataset_A_filenames[index]
        A = Image.open(filename_A).convert('RGB')

        filename_B = self.dataset_B_filenames[index]
        B = Image.open(filename_B).convert('RGB')

        A = A.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        B = B.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)

        A = self.transform(A)
        B = self.transform(B)
    
        return {'A': A, 'B': B,
                'A_paths': filename_A, 'B_paths': filename_B}

    def __len__(self):
        return len(self.dataset_A_filenames)

    def name(self):
        return 'AlignedDataset'
