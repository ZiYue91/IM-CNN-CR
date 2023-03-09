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


class SlopeNetModel(BaseModel):
    def name(self):
        return 'SlopeNetModel'


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


        self.netS.cuda(opt.gpu_ids[0])
        networks.init_weights(self.netS, 'normal')

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netS, 'S', opt.which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr

            self.criterionL2 = torch.nn.MSELoss()
            self.criterionL1 = torch.nn.L1Loss()

            self.schedulers = []
            self.optimizers = []
            self.optimizer_S = torch.optim.Adam(self.netS.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_S)

            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netS)

        print('-----------------------------------------------')

    def set_input(self, input):
        input_ori = input['A']

        #if self.isTrain:
        input_mask = input['mask']
        input_slope = input['slope']
        self.input_mask.resize_(input_mask.size()).copy_(input_mask)
        self.input_slope.resize_(input_slope.size()).copy_(input_slope)

        self.input_ori.resize_(input_ori.size()).copy_(input_ori)
        self.image_paths = input['A_paths']


    def forward(self):
        self.ori = Variable(self.input_ori, requires_grad=True)
        self.mask = Variable(torch.unsqueeze(self.input_mask, 1))
        slope = self.input_slope
        slope = torch.unsqueeze(slope, 2)
        slope = torch.unsqueeze(slope, 3)
        slope = torch.cat([slope,slope,slope,slope],2)
        slope = torch.cat([slope,slope,slope,slope],3)
        self.slope = Variable(slope)

        self.slope_proc = self.netS(self.mask, self.ori)


    def test(self):
        self.ori = Variable(self.input_ori, requires_grad=True)
        self.mask = Variable(torch.unsqueeze(self.input_mask, 1))
        slope = self.input_slope
        slope = torch.unsqueeze(slope, 2)
        slope = torch.unsqueeze(slope, 3)
        slope = torch.cat([slope,slope,slope,slope],2)
        slope = torch.cat([slope,slope,slope,slope],3)
        self.slope = Variable(slope)

        self.slope_proc = self.netS(self.mask, self.ori)


    def get_image_paths(self):
        return self.image_paths

    def backward_S(self):
        self.loss_S_reg = self.criterionL2(self.slope_proc, self.slope) * self.opt.lambda_A * 10
        self.loss_S_reg.backward(retain_graph=True)


    def optimize_parameters(self):
        self.forward()

        self.optimizer_S.zero_grad()
        self.backward_S()
        self.optimizer_S.step()


    def get_current_errors(self):
        return OrderedDict([('S_reg', self.loss_S_reg.item()),
                            ])

    def get_current_visuals(self):

        slope_543 = util.tensor2im11(self.slope.data[:, 0:3, :, :])
        slope_proc_543 = util.tensor2im11(self.slope_proc.data[:, 0:3, :, :])

        slope_321 = util.tensor2im11(self.slope.data[:, 2:5, :, :])
        slope_proc_321 = util.tensor2im11(self.slope_proc.data[:, 2:5, :, :])

        return OrderedDict([('slope_543', slope_543),('slope_proc_543', slope_proc_543),
                            ('slope_321', slope_321),('slope_proc_321', slope_proc_321),
                            ])

    def save(self, label):
        self.save_network(self.netS, 'S', label, self.gpu_ids)
