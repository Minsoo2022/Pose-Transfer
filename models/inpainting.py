import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .src import dp, grid_sampler
# losses
from losses.SegmentsStyleLoss import SegmentsSeperateStyleLoss

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
        self.input_P1, self.input_BP1 = input['P1'], input['BP1']
        self.input_P2, self.input_BP2 = input['P2'], input['BP2']

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
            #self.input_BBox1 = self.input_BBox1.cuda()
            #self.input_BBox2 = self.input_BBox1.cuda()


        source_mask = torch.logical_not(torch.isnan(self.input_BP1)).sum(dim=1) > 0
        target_mask = torch.logical_not(torch.isnan(self.input_BP2)).sum(dim=1) > 0
        self.source_mask = source_mask.float().unsqueeze(1)
        self.target_mask = target_mask.float().unsqueeze(1)


    def forward(self):
        # self.input_P1 = Variable(self.input_P1_set)
        # self.input_BP1 = Variable(self.input_BP1_set)
        #
        # self.input_P2 = Variable(self.input_P2_set)
        # self.input_BP2 = Variable(self.input_BP2_set)
        #
        # self.input_BBox1 = Variable(self.input_BBox1_set)
        # self.input_BBox2 = Variable(self.input_BBox2_set)
        source_uv = self.input_BP1.permute(0, 2, 3, 1)
        target_uv = self.input_BP2.permute(0, 2, 3, 1)
        source_img = self.input_P1


        # Replace NaNs with out-of-image coordinates
        source_uv[torch.isnan(source_uv)] = -10
        target_uv[torch.isnan(target_uv)] = -10

        # Get xy textures
        N, _, H, W = source_uv.shape
        meshgrid = make_meshgrid(self.H, self.W, device=self.device)
        meshgrid = torch.stack([meshgrid] * N, dim=0)

        source_xy_textures, _ = self.sampler(meshgrid, source_uv[..., [1, 0]])
        self.source_xy_textures= source_xy_textures

        source_fg = source_img * self.source_mask
        source_textures, source_holes = self.sampler(source_fg, source_uv[..., [1, 0]])

        source_holes = (source_holes[:, :1] > 1e-10).float()
        self.uv_mask=source_holes
        self.uv_texture= source_textures

        # Inpaint xy textures
        inp_in = torch.cat([source_xy_textures, source_holes[:, :1], meshgrid], dim=1)

        xy_inpainted = torch.tanh(self.netG(inp_in))
        self.xy_inpainted= xy_inpainted

        # Warp source image to get RGB textures
        pred_textures = F.grid_sample(source_fg, xy_inpainted.permute(0, 2, 3, 1))
        self.uv_texture_inpainted= pred_textures
        pred_img = F.grid_sample(pred_textures, target_uv)
        pred_img = pred_img * self.target_mask

        # G_input = [self.input_P1,
        #            torch.cat((self.input_BP1, self.input_BP2), 1),
        #            self.input_BBox1,
        #            self.input_BBox2]

        self.fake_p2 = pred_img

    def test(self):
        # self.input_P1 = Variable(self.input_P1_set)
        # self.input_BP1 = Variable(self.input_BP1_set)
        #
        # self.input_P2 = Variable(self.input_P2_set)
        # self.input_BP2 = Variable(self.input_BP2_set)
        #
        # self.input_BBox1 = Variable(self.input_BBox1_set)
        # self.input_BBox2 = Variable(self.input_BBox2_set)

        G_input = [self.input_P1,
                   torch.cat((self.input_BP1, self.input_BP2), 1),
                   self.input_BBox1,
                   self.input_BBox2]

        self.fake_p2 = self.netG(G_input)

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
        img_loss = self.criterionL1(self.fake_p2, self.input_P2 * self.target_mask)#, self.input_BBox2)
        coord_loss = self.criterionL1(self.uv_mask * self.source_xy_textures, self.uv_mask * self.xy_inpainted)  # , self.input_BBox2)
        pair_loss = img_loss+coord_loss
        pair_loss.backward()

        self.pair_L1loss = img_loss.item()
        self.coord_loss = coord_loss.item()
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
        ret_errors = OrderedDict([('pair_L1loss', self.pair_L1loss),
                                  ('coord_loss', self.coord_loss),
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
        input_xy = util.tensor2im(torch.cat([self.source_xy_textures.data,padding_zero],dim=1))
        inpainted_xy = util.tensor2im(torch.cat([self.xy_inpainted.data, padding_zero], dim=1))
        uv_texture = util.tensor2im(self.uv_texture.data)
        uv_texture_inpainted = util.tensor2im(self.uv_texture_inpainted.data)


        fake_p2 = util.tensor2im(self.fake_p2.data)

        tosave = [input_P1, input_BP1, input_xy, uv_texture, input_P2, input_BP2, inpainted_xy, uv_texture_inpainted, fake_p2]
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

        self.save_network(self.netD_PB, 'netD_PB', label, self.gpu_ids)
        if self.opt.with_D_PP:
            self.save_network(self.netD_PP, 'netD_PP', label, self.gpu_ids)

