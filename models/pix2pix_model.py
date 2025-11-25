import torch
from .base_model import BaseModel
from . import networks
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import numpy as np


class Pix2PixModel(BaseModel):
    """This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm="batch", netG="unet_256", dataset_mode="aligned")
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode="vanilla")
            parser.add_argument("--lambda_L1", type=float, default=100.0, help="weight for L1 loss")

        return parser

    # --- 新增: train.py 期望调用的公共接口 ---
    def get_current_metrics(self):
        """返回当前的 PSNR 和 SSIM 值 (以字典形式)"""
        return self._calculate_metrics()

    # --- 原 calculate_metrics 重命名为内部方法 ---
    def _calculate_metrics(self):
        """计算 PSNR 和 SSIM (原 calculate_metrics)"""
        
        # 安全检查
        if not hasattr(self, 'real_B') or self.real_B.shape[0] == 0:
            return {}
            
        # 将张量从 GPU 移动到 CPU，并转换为 NumPy 数组
        # 移除批次维度 (Batch dimension)
        # B 是真实图像 (Ground Truth)，fake_B 是生成图像
        real_B_np = self.real_B[0].cpu().detach().numpy().transpose(1, 2, 0)
        fake_B_np = self.fake_B[0].cpu().detach().numpy().transpose(1, 2, 0)

        # 我们的数据是归一化到 [-1, 1] 的浮点数。
        DATA_RANGE = 2.0
        
        # 1. PSNR (Peak Signal-to-Noise Ratio)
        # PSNR 要求输入是 HxW 或 HxWxC
        psnr_value = psnr(real_B_np, fake_B_np, data_range=DATA_RANGE)
        
        # 2. SSIM (Structural Similarity Index Measure)
        # 由于我们是单通道图像 (C=1)，我们先移除 C 维度
        if real_B_np.shape[-1] == 1:
            real_B_np = real_B_np.squeeze(-1) # H x W x 1 -> H x W
            fake_B_np = fake_B_np.squeeze(-1)

        ssim_value = ssim(
            real_B_np,  
            fake_B_np,  
            data_range=DATA_RANGE,
            multichannel=False # 确保对单通道图像进行正确计算
        )
        
        # 返回格式化后的结果
        return {'PSNR': float(f'{psnr_value:.4f}'), 'SSIM': float(f'{ssim_value:.4f}')}


    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ["G_GAN", "G_L1", "D_real", "D_fake"]
        
        # --- 修正: 确保 self.metric_names 始终被定义，以便可视化器识别 ---
        # 如果 train.py 调用 get_current_metrics，我们就需要定义这些名称
        self.metric_names = ['PSNR', 'SSIM'] 
        # ------------
            
        self.visual_names = ["real_A", "fake_B", "real_B"]
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ["G", "D"]
        else:  # during test time, only load G
            self.model_names = ["G"]
        self.device = opt.device
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # move to the device for custom loss
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # update G's weights
