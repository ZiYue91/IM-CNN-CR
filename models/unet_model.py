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


class UNetModel(BaseModel):
    def name(self):
        return 'UNetModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.output_nc = opt.output_nc
        gpu_ids = opt.gpu_ids
        norm_layer = functools.partial(torch.nn.BatchNorm2d, affine=True)

        self.input_ori = torch.cuda.FloatTensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_gt = torch.cuda.FloatTensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_mask = torch.cuda.FloatTensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)

        self.netT = nets.UNet(opt)


        self.netT.cuda(opt.gpu_ids[0])
        networks.init_weights(self.netT, 'normal')

        if not self.isTrain or opt.continue_train:           
            self.load_network(self.netT, 'T', opt.which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr

            self.criterionL2 = torch.nn.MSELoss()
            self.criterionL1 = torch.nn.L1Loss()

            self.schedulers = []
            self.optimizers = []
            self.optimizer_T = torch.optim.Adam(self.netT.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_T)

            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netT)

        print('-----------------------------------------------')

    def set_input(self, input):
        input_ori = input['A']

        #if self.isTrain:
        input_mask = input['mask']
        self.input_mask.resize_(input_mask.size()).copy_(input_mask)

        self.input_ori.resize_(input_ori.size()).copy_(input_ori)
        self.image_paths = input['A_paths']


    def forward(self):
        self.ori = Variable(self.input_ori, requires_grad=True)
        self.mask = Variable(torch.unsqueeze(self.input_mask, 1))

        self.thicknessmap = self.netT(self.ori)


    def test(self):
        self.ori = Variable(self.input_ori)
        self.mask = Variable(torch.unsqueeze(self.input_mask, 1))

        self.thicknessmap = self.netT(self.ori)


    def get_image_paths(self):
        return self.image_paths

    def backward_T(self):
        self.loss_tmap = self.criterionL2(self.thicknessmap, self.mask) * self.opt.lambda_A * 10
        self.loss_tmap.backward(retain_graph=True)

    def optimize_parameters(self):
        self.forward()

        self.optimizer_T.zero_grad()
        self.backward_T()
        self.optimizer_T.step()

    def get_current_errors(self):
        return OrderedDict([('T_reg', self.loss_tmap.item()),
                            ])

    def get_current_visuals(self):
        thicknessmap = util.tensor2im01(self.thicknessmap.data)
        mask = util.tensor2im01(self.mask.data)

        return OrderedDict([('thicknessmap', thicknessmap), ('mask', mask)
                            ])

    def save(self, label):
        self.save_network(self.netT, 'T', label, self.gpu_ids)
