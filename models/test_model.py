import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import nets
import functools


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.output_nc = opt.output_nc
        self.thres = 0.2
        gpu_ids = opt.gpu_ids
        norm_layer = functools.partial(torch.nn.BatchNorm2d, affine=True)

        self.input_ori = torch.cuda.FloatTensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_gt = torch.cuda.FloatTensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_mask = torch.cuda.FloatTensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
        self.input_slope = torch.cuda.FloatTensor(opt.batchSize, opt.input_nc, 1, 1)

        self.netS = nets.SlopeNet(opt)
        self.netT = nets.UNet(opt)

        self.netT.cuda(opt.gpu_ids[0])
        networks.init_weights(self.netT, 'normal')

        self.netS.cuda(opt.gpu_ids[0])
        networks.init_weights(self.netS, 'normal')

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netS, 'S', opt.which_epoch)
            self.load_network(self.netT, 'T', opt.which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr

            self.criterionL2 = torch.nn.MSELoss()
            self.criterionL1 = torch.nn.L1Loss()

            self.schedulers = []
            self.optimizers = []
            self.optimizer_T = torch.optim.Adam(self.netT.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_S = torch.optim.Adam(self.netS.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_T)
            self.optimizers.append(self.optimizer_S)

            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netT)
        networks.print_network(self.netS)

        print('-----------------------------------------------')

    def set_input(self, input):
        input_ori = input['A']

        # if self.isTrain:
        input_gt = input['B']
        input_mask = input['mask']
        input_slope = input['slope']
        self.input_mask.resize_(input_mask.size()).copy_(input_mask)
        self.input_slope.resize_(input_slope.size()).copy_(input_slope)
        self.input_gt.resize_(input_gt.size()).copy_(input_gt)

        self.input_ori.resize_(input_ori.size()).copy_(input_ori)
        self.image_paths = input['A_paths']

    def forward(self):
        self.ori = Variable(self.input_ori, requires_grad=True)
        self.gt = Variable(self.input_gt)
        self.mask = Variable(torch.unsqueeze(self.input_mask, 1))
        slope = self.input_slope
        slope = torch.unsqueeze(slope, 2)
        slope = torch.unsqueeze(slope, 3)
        slope = torch.cat([slope, slope, slope, slope], 2)
        slope = torch.cat([slope, slope, slope, slope], 3)
        self.slope = Variable(slope)

        self.thicknessmap = self.netT(self.ori)
        self.slope_proc = self.netS(self.thicknessmap, self.ori)
        slope_proc1 = torch.mean(self.slope_proc, dim=2, keepdim=True)
        slope_proc1 = torch.mean(slope_proc1, dim=3, keepdim=True)
        self.proc = self.ori - self.thicknessmap * slope_proc1

    def test(self):
        self.ori = Variable(self.input_ori, requires_grad=True)
        self.gt = Variable(self.input_gt)
        self.mask = Variable(torch.unsqueeze(self.input_mask, 1))
        slope = self.input_slope
        slope = torch.unsqueeze(slope, 2)
        slope = torch.unsqueeze(slope, 3)
        slope = torch.cat([slope, slope, slope, slope], 2)
        slope = torch.cat([slope, slope, slope, slope], 3)
        self.slope = Variable(slope)

        self.thicknessmap = self.netT(self.ori)
        self.slope_proc = self.netS(self.thicknessmap, self.ori)
        slope_proc1 = torch.mean(self.slope_proc, dim=2, keepdim=True)
        slope_proc1 = torch.mean(slope_proc1, dim=3, keepdim=True)
        self.proc = self.ori - self.thicknessmap * slope_proc1

    def get_image_paths(self):
        return self.image_paths


    def get_current_visuals(self):

        thicknessmap = util.tensor2im01(self.thicknessmap.data)
        mask = util.tensor2im01(self.mask.data)

        slope_543 = util.tensor2im11(self.slope.data[:, 0:3, :, :])
        slope_proc_543 = util.tensor2im11(self.slope_proc.data[:, 0:3, :, :])

        slope_321 = util.tensor2im11(self.slope.data[:, 2:5, :, :])
        slope_proc_321 = util.tensor2im11(self.slope_proc.data[:, 2:5, :, :])

        proc_543 = util.tensor2im01(self.proc.data[:, 0:3, :, :])
        ori_543 = util.tensor2im01(self.ori.data[:, 0:3, :, :])
        gt_543 = util.tensor2im01(self.gt.data[:, 0:3, :, :])

        proc_321 = util.tensor2im01(self.proc.data[:, 2:5, :, :])
        ori_321 = util.tensor2im01(self.ori.data[:, 2:5, :, :])
        gt_321 = util.tensor2im01(self.gt.data[:, 2:5, :, :])

        return OrderedDict(
            [('thicknessmap', thicknessmap), ('mask', mask),
             ('slope_543', slope_543), ('slope_proc_543', slope_proc_543),
             ('slope_321', slope_321), ('slope_proc_321', slope_proc_321),
             ('proc_543', proc_543), ('ori_543', ori_543), ('gt_543', gt_543),
             ('proc_321', proc_321), ('ori_321', ori_321), ('gt_321', gt_321),
             ])

    def save(self, label):
        self.save_network(self.netT, 'T', label, self.gpu_ids)
        self.save_network(self.netS, 'S', label, self.gpu_ids)
