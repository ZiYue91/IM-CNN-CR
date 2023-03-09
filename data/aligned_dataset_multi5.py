import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np

class AlignedDatasetMulti5(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.A = torch.FloatTensor(opt.input_nc, opt.fineSize, opt.fineSize)
        self.B = torch.FloatTensor(opt.input_nc, opt.fineSize, opt.fineSize)

        self.ipath432 = os.path.join(self.root, 'simulate/432')
        self.ipath951 = os.path.join(self.root, 'simulate/951')
        self.ipathtxt = os.path.join(self.root, 'simulate/txt')


        self.i432names = sorted(make_dataset(self.ipath432))
        self.i951names = sorted(make_dataset(self.ipath951))
        self.itxtnames = sorted(make_dataset(self.ipathtxt))

        self.opath432 = os.path.join(self.root, 'real/432')
        self.o432names = sorted(make_dataset(self.opath432))
        self.opath951 = os.path.join(self.root, 'real/951')
        self.o951names = sorted(make_dataset(self.opath951))

        transform_list = [transforms.ToTensor()]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        i432 = self.i432names[index]
        iA = Image.open(i432).convert('RGB')
        i951 = self.i951names[index]
        iC = Image.open(i951).convert('RGB')
        itxt = self.itxtnames[index]
        iD = np.loadtxt(itxt)

        o432 = self.o432names[index]
        oA = Image.open(o432).convert('RGB')
        o951 = self.o951names[index]
        oC = Image.open(o951).convert('RGB')

        iA = iA.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        iC = iC.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        iA = self.transform(iA)  # 432
        iC = self.transform(iC)  # 951

        oA = oA.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        oC = oC.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        oA = self.transform(oA)
        oC = self.transform(oC)

        self.A[0,:,:].copy_(iC[1,:,:])
        self.A[1:4,:,:].copy_(iA)
        self.A[4,:,:].copy_(iC[2,:,:])

        self.B[0, :, :].copy_(oC[1, :, :])
        self.B[1:4, :, :].copy_(oA)
        self.B[4, :, :].copy_(oC[2, :, :])

        mask = iC[0,:,:]

        slope = torch.FloatTensor(iD)

        return {'A': self.A, 'B': self.B, 'mask': mask, 'slope': slope,
                'A_paths': i432, 'B_paths': o432, 'C_paths': i432}

    def __len__(self):
        return len(self.i432names)

    def name(self):
        return 'AlignedDatasetMulti5'
