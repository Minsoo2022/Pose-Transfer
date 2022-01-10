import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Resize
from PIL import Image
from collections import OrderedDict
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .src import dp, grid_sampler
import wandb
import os
# losses
from losses.SegmentsStyleLoss import SegmentsSeperateStyleLoss

def get_row(coor, num, masks):
    results = []
    temp = coor.reshape(-1, num, num, 2)
    sub1 = (temp[:, :, 1:] - temp[:, :, :-1]) ** 2
    sub2 = torch.abs(sub1[:, :, 1:] - sub1[:, :, :-1])
    for mask in masks:
        sub2_ = sub2 * (mask[:,:,1:-1][..., None].repeat(1,1,1,2))
        sub2_ = sub2_.reshape(-1, num * (num - 2), 2)
        results.append(sub2_)
    return results

def get_col(coor, num, masks):
    results = []
    temp = coor.reshape(-1, num, num, 2)
    sub1 = (temp[:, 1:, :] - temp[:, :-1, :]) ** 2
    sub2 = torch.abs(sub1[:, 1:, :] - sub1[:, :-1, :])
    for mask in masks:
        sub2_ = sub2 * (mask[:, 1:-1, :][..., None].repeat(1, 1, 1, 2))
        sub2_ = sub2_.permute(0, 2, 1, 3).reshape(-1, num * (num - 2), 2)
        results.append(sub2_)
    return results

def grad_row(coor, num, masks):
    results = []
    temp = coor.reshape([-1, num, num, 2])
    sub1 = (temp[:, :, 1:-1] - temp[:, :, :-2])
    sub2 = (temp[:, :, 1:-1] - temp[:, :, 2:])
    sub3 = torch.abs(sub1[:, :, :, 1] * sub2[:, :, :, 0] - sub2[:, :, :, 1] * sub1[:, :, :, 0])
    for mask in masks:
        sub3_ = sub3 * (mask[:, :, 1:-1])
        sub3_ = sub3_.reshape(-1, num * (num - 2))
        results.append(sub3_)
    return results # sub3.sum(1).mean()

def grad_col(coor, num, masks):
    results = []
    temp = coor.reshape([-1, num, num, 2])
    sub1 = (temp[:, 1:-1, :] - temp[:, :-2, :])
    sub2 = (temp[:, 1:-1, :] - temp[:, 2:, :])
    sub3 = torch.abs(sub1[:, :, :, 1] * sub2[:, :, :, 0] - sub2[:, :, :, 1] * sub1[:, :, :, 0])
    for mask in masks:
        sub3_ = sub3 * (mask[:, 1:-1, :])
        sub3_ = sub3_.reshape(-1, num * (num - 2))
        results.append(sub3_)
    return results # sub3.sum(1).mean()

def make_meshgrid(H, W, device='cuda:0'):
    x = torch.arange(0, W).to(device)
    y = torch.arange(0, H).to(device)

    xx, yy = torch.meshgrid([x, y])
    meshgrid = torch.stack([yy, xx], dim=0).float()

    meshgrid[0] = (2 * meshgrid[0] / (H - 1)) - 1
    meshgrid[1] = (2 * meshgrid[1] / (W - 1)) - 1

    return meshgrid

class DeformablePipe(BaseModel):
    def name(self):
        return 'DeformablePipe'

    def initialize(self, opt):
        self.H = 256 # 나중에 opt에서 받도록
        self.W = 256
        BaseModel.initialize(self, opt)
        self.phase = opt.phase
        # self.input_P1_set = self.Tensor(nb, opt.P_input_nc, size, size)
        # self.input_BP1_set = self.Tensor(nb, opt.BP_input_nc, size, size)
        # self.input_P2_set = self.Tensor(nb, opt.P_input_nc, size, size)
        # self.input_BP2_set = self.Tensor(nb, opt.BP_input_nc, size, size)
        # self.input_BBox1_set = self.Tensor(nb, opt.nsegments, 4)
        # self.input_BBox2_set = self.Tensor(nb, opt.nsegments, 4)
        self.num_of_Goutput = 1
        self.device = f"cuda:{self.gpu_ids[0]}"
        input_nc = [opt.P_input_nc, opt.BP_input_nc + opt.BP_input_nc]
        self.netG = dp.GatedHourglass(32, 5, 2).cuda(self.gpu_ids[0])
        self.sampler = grid_sampler.InvGridSamplerDecomposed(return_B=True, hole_fill_color=0.).cuda(self.gpu_ids[0])

        self.masks = []
        to_tensor = ToTensor()
        resize = Resize(256, interpolation=0)
        for i in range(1, 7):
            mask = to_tensor(resize(Image.open(f'./masks/mask{i}.png')))[:1]
            self.masks.append(mask.cuda(self.device))
        # self.netG = networks.define_G(input_nc, opt.P_input_nc,
        #                               opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type,
        #                               self.gpu_ids,
        #                               # num. of down-sampling blocks
        #                               n_downsampling=opt.G_n_downsampling,
        #                               norm_affine=opt.norm_affine
        #                               )

        # if self.isTrain:
        #     use_sigmoid = opt.no_lsgan
        #     self.netD_PB = networks.define_D(opt.P_input_nc + opt.BP_input_nc, opt.ndf,
        #                                      opt.which_model_netD,
        #                                      opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
        #                                      # for resD, dropout is selective.
        #                                      not opt.no_dropout_D,
        #                                      n_downsampling=opt.D_n_downsampling)
        #
        #     if opt.with_D_PP:
        #         self.netD_PP = networks.define_D(opt.P_input_nc + opt.P_input_nc, opt.ndf,
        #                                          opt.which_model_netD,
        #                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
        #                                          # for resD, dropout is selective.
        #                                          not opt.no_dropout_D,
        #                                          n_downsampling=opt.D_n_downsampling)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'netG', which_epoch)
            # if self.isTrain:
            #     self.load_network(self.netD_PB, 'netD_PB', which_epoch)
            #     if opt.with_D_PP:
            #         self.load_network(self.netD_PP, 'netD_PP', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_PP_pool = ImagePool(opt.pool_size)
            self.fake_PB_pool = ImagePool(opt.pool_size)
            # define loss functions
            # self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            #self.criterionL1 = SegmentsSeperateStyleLoss(opt.nsegments, opt.lambda_A, opt.lambda_B, opt.lambda_style,
            #                                             opt.perceptual_layers, self.gpu_ids[0])
            self.criterionL1= torch.nn.L1Loss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_D_PB = torch.optim.Adam(filter(lambda p: p.requires_grad, self.netD_PB.parameters()),
            #                                        lr=opt.lr, betas=(opt.beta1, 0.999))
            #
            # if opt.with_D_PP:
            #     self.optimizer_D_PP = torch.optim.Adam(filter(lambda p: p.requires_grad, self.netD_PP.parameters()),
            #                                            lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            # self.optimizers.append(self.optimizer_D_PB)
            # if opt.with_D_PP:
            #     self.optimizers.append(self.optimizer_D_PP)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        # if self.isTrain:
        #     networks.print_network(self.netD_PB)
        #     if opt.with_D_PP:
        #         networks.print_network(self.netD_PP)
        print('-----------------------------------------------')

    def set_input(self, input):
        self.input_P1, self.input_BP1, self.input_BP1_flip = input['P1'], input['BP1'], input['BP1_flip']
        self.input_P2, self.input_BP2, self.input_BP2_flip = input['P2'], input['BP2'], input['BP2_flip']

        # self.input_P1_set.resize_(input_P1.size()).copy_(input_P1)
        # self.input_BP1_set.resize_(input_BP1.size()).copy_(input_BP1)
        # self.input_P2_set.resize_(input_P2.size()).copy_(input_P2)
        # self.input_BP2_set.resize_(input_BP2.size()).copy_(input_BP2)

        #self.input_BBox1, self.input_BBox2 = input['BBox1'], input['BBox2']
        # self.input_BBox1_set.resize_(input_BBox1.size()).copy_(input_BBox1)
        # self.input_BBox2_set.resize_(input_BBox2.size()).copy_(input_BBox2)

        self.image_paths = input['P1_path'][0] + '___' + input['P2_path'][0]

        if len(self.gpu_ids) > 0:
            self.input_P1 = self.input_P1.cuda()
            self.input_BP1 = self.input_BP1.cuda()
            self.input_P2 = self.input_P2.cuda()
            self.input_BP2 = self.input_BP2.cuda()
            self.input_BP1_flip = self.input_BP1_flip.cuda()
            self.input_BP2_flip = self.input_BP2_flip.cuda()
            #self.input_BBox1 = self.input_BBox1.cuda()
            #self.input_BBox2 = self.input_BBox1.cuda()


        source_mask = torch.logical_not(torch.isnan(self.input_BP1)).sum(dim=1) > 0
        target_mask = torch.logical_not(torch.isnan(self.input_BP2)).sum(dim=1) > 0

        self.source_xy_mask = source_mask.float().unsqueeze(1)
        self.target_xy_mask = target_mask.float().unsqueeze(1)


    def forward(self):
        # self.input_P1 = Variable(self.input_P1_set)
        # self.input_BP1 = Variable(self.input_BP1_set)
        #
        # self.input_P2 = Variable(self.input_P2_set)
        # self.input_BP2 = Variable(self.input_BP2_set)
        #
        # self.input_BBox1 = Variable(self.input_BBox1_set)
        # self.input_BBox2 = Variable(self.input_BBox2_set)

        # source_uv -> source_xy_uv
        # target_uv -> target_xy_uv
        # source_img -> source_xy_texture
        # source_holes -> source_uv_mask
        # source_xy_textures -> source_uv_xy
        # pred_textures -> uv_texture_inpainted
        # pred_img -> target_xy_texture_pred


        source_xy_uv = self.input_BP1.permute(0, 2, 3, 1)
        source_xy_uv_flip = self.input_BP1_flip.permute(0, 2, 3, 1)
        target_xy_uv = self.input_BP2.permute(0, 2, 3, 1)

        source_xy_texture = self.input_P1

        # Replace NaNs with out-of-image coordinates
        source_xy_uv[torch.isnan(source_xy_uv)] = -10
        target_xy_uv[torch.isnan(target_xy_uv)] = -10
        source_xy_uv_flip[torch.isnan(source_xy_uv_flip)] = -10

        # Get xy textures
        N, _, H, W = source_xy_uv.shape
        meshgrid = make_meshgrid(self.H, self.W, device=self.device)
        meshgrid = torch.stack([meshgrid] * N, dim=0)

        self.source_uv_xy, source_uv_mask = self.sampler(meshgrid, source_xy_uv[..., [1, 0]])
        self.source_uv_mask = (source_uv_mask[:, :1] > 1e-10).float()

        self.source_uv_xy_flip, source_uv_mask_flip = self.sampler(meshgrid, source_xy_uv_flip[..., [1, 0]])
        source_uv_mask_flip = (source_uv_mask_flip[:, :1] > 1e-10).float()

        source_uv_mask_flip = source_uv_mask_flip.int() - (self.source_uv_mask * source_uv_mask_flip).int()
        self.source_uv_mask_flip = source_uv_mask_flip.bool()
        self.source_uv_mask_union = self.source_uv_mask + self.source_uv_mask_flip

        self.source_uv_xy_union = self.source_uv_xy * self.source_uv_mask + self.source_uv_xy_flip * self.source_uv_mask_flip

        source_xy_texture_fg = source_xy_texture * self.source_xy_mask
        self.source_uv_texture , _ = self.sampler(source_xy_texture_fg, source_xy_uv[..., [1, 0]])

        # Inpaint xy textures
        # inp_in = torch.cat([source_uv_xy, source_uv_mask[:, :1], meshgrid], dim=1)
        inp_in = torch.cat([self.source_uv_xy_union, self.source_uv_mask_union[:, :1], meshgrid], dim=1)

        self.uv_xy_inpainted = torch.tanh(self.netG(inp_in))

        # Warp source image to get RGB textures
        #self.uv_texture_inpainted  = F.grid_sample(source_xy_texture_fg, self.uv_xy_inpainted.permute(0, 2, 3, 1))
        self.uv_texture_inpainted = F.grid_sample(source_xy_texture, self.uv_xy_inpainted.permute(0, 2, 3, 1))

        self.fake_p2 = F.grid_sample(self.uv_texture_inpainted , target_xy_uv)
        self.fake_p1 = F.grid_sample(self.uv_texture_inpainted, source_xy_uv)


        ###
        # target_xy_uv_flip = self.input_BP2_flip.permute(0, 2, 3, 1)
        # target_xy_uv_flip[torch.isnan(target_xy_uv_flip)] = -10
        # self.fake_p1_flip = F.grid_sample(self.uv_texture_inpainted , source_xy_uv_flip)
        # self.fake_p2_flip = F.grid_sample(self.uv_texture_inpainted , target_xy_uv_flip)
        self.source_uv_texture_union = F.grid_sample(source_xy_texture_fg, self.source_uv_xy_union.permute(0, 2, 3, 1)) * self.source_uv_mask_union
        self.source_uv_texture_flip = F.grid_sample(source_xy_texture_fg, self.source_uv_xy_flip.permute(0, 2, 3, 1)) * self.source_uv_mask_flip

    def validate(self):
        with torch.no_grad():
            self.forward()
    def test(self):
        with torch.no_grad():
            self.forward()
    #     # self.input_P1 = Variable(self.input_P1_set)
    #     # self.input_BP1 = Variable(self.input_BP1_set)
    #     #
    #     # self.input_P2 = Variable(self.input_P2_set)
    #     # self.input_BP2 = Variable(self.input_BP2_set)
    #     #
    #     # self.input_BBox1 = Variable(self.input_BBox1_set)
    #     # self.input_BBox2 = Variable(self.input_BBox2_set)
    #
    #     G_input = [self.input_P1,
    #                torch.cat((self.input_BP1, self.input_BP2), 1),
    #                self.input_BBox1,
    #                self.input_BBox2]
    #
    #     self.fake_p2 = self.netG(G_input)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_unpaired_G(self):

        # if self.opt.with_D_PB:
        #     pred_fake_PB = self.netD_PB(torch.cat((self.fake_p2, self.input_BP2), 1))
        #     self.loss_G_GAN_PB = self.criterionGAN(pred_fake_PB, True)
        #
        # if self.opt.with_D_PP:
        #     pred_fake_PP = self.netD_PP(torch.cat((self.fake_p2, self.input_P1), 1))
        #     self.loss_G_GAN_PP = self.criterionGAN(pred_fake_PP, True)
        #
        # if self.opt.with_D_PB:
        #     pair_GANloss = self.loss_G_GAN_PB * self.opt.lambda_GAN
        #     if self.opt.with_D_PP:
        #         pair_GANloss += self.loss_G_GAN_PP * self.opt.lambda_GAN
        #         pair_GANloss = pair_GANloss / 2
        # else:
        #     if self.opt.with_D_PP:
        #         pair_GANloss = self.loss_G_GAN_PP * self.opt.lambda_GAN

        # L1 loss
        img_loss_target = self.criterionL1(self.fake_p2 * self.target_xy_mask, self.input_P2 * self.target_xy_mask) # + 0.5 * self.criterionL1(self.fake_p2_flip * self.target_xy_mask, self.input_P2 * self.target_xy_mask)
        img_loss_source = self.criterionL1(self.fake_p1 * self.source_xy_mask, self.input_P1 * self.source_xy_mask) # + 0.5 * self.criterionL1(self.fake_p1_flip * self.target_xy_mask, self.input_P2 * self.target_xy_mask)
        # coord_loss = self.criterionL1(self.source_uv_mask * self.source_uv_xy, self.source_uv_mask * self.uv_xy_inpainted)  # , self.input_BBox2)
        # coord_loss = self.criterionL1(self.source_uv_mask_union * self.source_uv_xy_union, self.source_uv_mask_union * self.uv_xy_inpainted)  # , self.input_BBox2)
        coord_loss = self.criterionL1(self.source_uv_mask * self.source_uv_xy, self.source_uv_mask * self.uv_xy_inpainted) \
                     + 0.5 * self.criterionL1(self.source_uv_mask_flip * self.source_uv_xy_flip, self.source_uv_mask_flip * self.uv_xy_inpainted)


        # coor = theta.view(inputA.shape[0], -1, 2)
        if self.opt.lambda_source > 0:
            coor = self.uv_xy_inpainted.permute(0,2,3,1)
            coor = coor.reshape(coor.shape[0], coor.shape[1] * coor.shape[2], coor.shape[3])
            row = get_row(coor, 256, self.masks)
            row = torch.stack(row).sum(dim=0)
            col = get_col(coor, 256, self.masks)
            col = torch.stack(col).sum(dim=0)
            rg_loss = grad_row(coor, 256, self.masks)
            rg_loss = torch.stack(rg_loss).sum(dim=0).sum(dim=1).mean()
            cg_loss = grad_col(coor, 256, self.masks)
            cg_loss = torch.stack(cg_loss).sum(dim=0).sum(dim=1).mean()
            rg_loss = torch.max(rg_loss, torch.tensor(1.).cuda())
            cg_loss = torch.max(cg_loss, torch.tensor(1.).cuda())
            rx, ry, cx, cy = torch.tensor(0.08).cuda(), torch.tensor(0.08).cuda() \
                , torch.tensor(0.08).cuda(), torch.tensor(0.08).cuda()
            row_x, row_y = row[:, :, 0], row[:, :, 1]
            col_x, col_y = col[:, :, 0], col[:, :, 1]
            rx_loss = torch.max(rx, row_x).mean()
            ry_loss = torch.max(ry, row_y).mean()
            cx_loss = torch.max(cx, col_x).mean()
            cy_loss = torch.max(cy, col_y).mean()
            constraint_loss = rg_loss + cg_loss + rx_loss + ry_loss + cx_loss + cy_loss
        else:
            constraint_loss = 0

        pair_loss = img_loss_target * self.opt.lambda_target + img_loss_source * self.opt.lambda_source + \
                    coord_loss * self.opt.lambda_coord + constraint_loss * self.opt.lambda_constraint
        pair_loss.backward()

        # self.pair_L1loss = img_loss.item()
        self.img_target_L1loss = img_loss_target.item()
        self.img_source_L1loss = img_loss_source.item()
        self.coord_loss = coord_loss.item()
        self.constraint_loss = constraint_loss.item()
        # self.pair_GANloss = pair_GANloss.item()

    # def backward_D_basic(self, netD, real, fake):
    #     # Real
    #     pred_real = netD(real)
    #     loss_D_real = self.criterionGAN(pred_real, True)
    #     # Fake
    #     pred_fake = netD(fake.detach())
    #     loss_D_fake = self.criterionGAN(pred_fake, False)
    #     # Combined loss
    #     loss_D = (loss_D_real + loss_D_fake) * 0.5
    #     # backward
    #     loss_D.backward()
    #     return loss_D
    #
    # # D: take(P, B) as input
    # def backward_D_PB(self):
    #     real_PB = torch.cat((self.input_P2, self.input_BP2), 1)
    #     fake_PB = self.fake_PB_pool.query(torch.cat((self.fake_p2, self.input_BP2), 1))
    #     loss_D_PB = self.backward_D_basic(self.netD_PB, real_PB, fake_PB)
    #     self.loss_D_PB = loss_D_PB.item()
    #
    # # D: take(P, P') as input
    # def backward_D_PP(self):
    #     real_PP = torch.cat((self.input_P2, self.input_P1), 1)
    #     fake_PP = self.fake_PP_pool.query(torch.cat((self.fake_p2, self.input_P1), 1))
    #     loss_D_PP = self.backward_D_basic(self.netD_PP, real_PP, fake_PP)
    #     self.loss_D_PP = loss_D_PP.item()

    def optimize_parameters(self):
        # forward
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_unpaired_G()
        self.optimizer_G.step()

        # # D_P
        # if self.opt.with_D_PP:
        #     for i in range(self.opt.DG_ratio):
        #         self.optimizer_D_PP.zero_grad()
        #         self.backward_D_PP()
        #         self.optimizer_D_PP.step()
        #
        # # D_BP
        # for i in range(self.opt.DG_ratio):
        #     self.optimizer_D_PB.zero_grad()
        #     self.backward_D_PB()
        #     self.optimizer_D_PB.step()

    def get_current_errors(self):
        # ('pair_L1loss', self.pair_L1loss),
        ret_errors = OrderedDict([('img_target_L1loss', self.img_target_L1loss),
                                  ('img_source_L1loss', self.img_source_L1loss),
                                  ('coord_loss', self.coord_loss),
                                  ('constraint_loss', self.constraint_loss)
                                  ])

        # ret_errors['origin_L1'] = self.loss_originL1
        # ret_errors['perceptual'] = self.loss_perceptual
        # ret_errors['style'] = self.loss_style

        return ret_errors

    def get_current_visuals(self):
        height, width = self.input_P1.size(2), self.input_P1.size(3)
        input_P1 = util.tensor2im(self.input_P1.data)
        input_P2 = util.tensor2im(self.input_P2.data)

        # input_BP1 = util.draw_pose_from_map(self.input_BP1.data)[0]
        # input_BP2 = util.draw_pose_from_map(self.input_BP2.data)[0]

        padding_zero =torch.zeros_like(self.input_BP1.data)[:,:1,:,:]

        input_BP1 = util.tensor2im(torch.cat([self.input_BP1.data,padding_zero],dim=1))
        input_BP2 = util.tensor2im(torch.cat([self.input_BP2.data,padding_zero],dim=1))
        source_uv_xy = util.tensor2im(torch.cat([self.source_uv_xy.data, padding_zero],dim=1))
        source_uv_xy_flip = util.tensor2im(torch.cat([self.source_uv_xy_flip.data, padding_zero], dim=1))
        source_uv_xy_union = util.tensor2im(torch.cat([self.source_uv_xy_union.data, padding_zero], dim=1))
        uv_xy_inpainted = util.tensor2im(torch.cat([self.uv_xy_inpainted.data, padding_zero], dim=1))
        source_uv_texture = util.tensor2im(self.source_uv_texture.data)
        uv_texture_inpainted = util.tensor2im(self.uv_texture_inpainted.data)

        source_uv_texture_union = util.tensor2im(self.source_uv_texture_union.data)
        source_uv_texture_flip = util.tensor2im(self.source_uv_texture_flip.data)

        fake_p1 = util.tensor2im(self.fake_p1.data)
        fake_p2 = util.tensor2im(self.fake_p2.data)

        tosave = [input_P1, source_uv_xy, source_uv_texture, source_uv_xy_flip, source_uv_texture_flip, source_uv_xy_union, source_uv_texture_union,

                  uv_xy_inpainted, uv_texture_inpainted, input_BP1, fake_p1,input_BP2,fake_p2,  input_P2]



        # nrow = 2
        # vis = np.zeros((height * nrow, width * len(tosave), 3)).astype(np.uint8)
        # for i in range(nrow):
        #     for j in range(len(tosave) % nrow):
        #         vis[i*height: (i+1)*height, j*width:(j+1)*width, :] = tosave[i*nrow + j]

        vis = np.zeros((height, width * len(tosave), 3)).astype(np.uint8)
        for i in range(len(tosave)):
            vis[:, i*width:(i+1)*width, :] = tosave[i]

        # vis = np.zeros((height, width * 5, 3)).astype(np.uint8)  # h, w, c
        # vis[:, :width, :] = input_P1
        # vis[:, width:width * 2, :] = input_BP1
        # vis[:, width * 2:width * 3, :] = input_P2
        # vis[:, width * 3:width * 4, :] = input_BP2
        # vis[:, width * 4:, :] = fake_p2

        ret_visuals = OrderedDict([('vis', vis)])

        return ret_visuals

    def save(self, label):
        self.save_network(self.netG, 'netG', label, self.gpu_ids)
        #
        # self.save_network(self.netD_PB, 'netD_PB', label, self.gpu_ids)
        # if self.opt.with_D_PP:
        #     self.save_network(self.netD_PP, 'netD_PP', label, self.gpu_ids)

    def init_wandb(self,):
        self.use_wandb= True
        if self.use_wandb:
            config_dict = {}
            # for key, value in self.configs.items():
            #     config_dict[str(key)] = value
            id = wandb.util.generate_id()
            with open(self.opt.checkpoints_dir+"/"+self.opt.name+"/wandb_id.txt", 'w') as file:
                file.write(id)

            config_dict['wandb_id'] = id
            config_dict['wandb_run_name'] = self.opt.name
            wandb.init(
                entity='fasker_research',  # davian-nerf',
                project='inpainting_stage1',  # project name
                id=id,
                resume="allow",
                config=config_dict,
            )
            wandb.run.name = self.opt.name
            # resume_path = ""
            # if len(self.configs['experiment']['resume_folder']) > 0:
            #     resume_path = Path(f"{self.configs['experiment']['exp_dir']}").parent
            #     resume_path = f"{resume_path}/{self.configs['experiment']['resume_folder']}/wandb_id.txt"
            #
            # if len(resume_path) > 0 and Path(f"{resume_path}").exists():
            #     print(f"Resume experiment from {resume_path}")
            #     with open(f"{resume_path}", 'r') as file:
            #         id = file.readline()
            # else:
            #     id = wandb.util.generate_id()


            # ### Resume from experiment without wandb_id
            # if (len(self.configs['experiment']['resume_folder']) > 0 and not Path(f"{self.configs['experiment']['resume_folder']}/wandb_id.txt").exists()):
            #     wandb.run.name = f"{self.configs['wandb']['wandb_run_name']}_(resume{self.configs['experiment']['resume_folder']})"
            # elif self.configs['wandb']['wandb_run_name'] is not None:   ##Start new experiment
            #         wandb.run.name = self.configs['wandb']['wandb_run_name']

