import torch
from .base_model import BaseModel
from . import networks
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import numpy as np
# 导入 Masked PSNR 计算函数 (保留你之前的实用工具)
from util.util import calculate_masked_psnr


class GradientLoss(torch.nn.Module):
    """
    梯度差分损失 (Gradient Difference Loss, GDL)
    
    原理：
    不仅比较像素值 (L1)，还比较像素的变化率 (梯度)。
    CT 图像中骨骼边缘梯度变化剧烈，这个 Loss 能强迫模型生成锐利的边缘，
    防止 L1 Loss 导致的模糊 (Regression-to-Mean)。
    """
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.criterion = torch.nn.L1Loss()

    def forward(self, x, y):
        # 1. 计算 X 轴方向的梯度 (右边像素 - 左边像素)
        # x[:, :, :, :-1] 表示不取最后一列
        # x[:, :, :, 1:]  表示不取第一列
        # 两者相减就是相邻像素的差值
        g_x_x = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        g_y_x = torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])
        
        # 2. 计算 Y 轴方向的梯度 (下边像素 - 上边像素)
        g_x_y = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        g_y_y = torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])
        
        # 3. 计算生成图梯度和真实图梯度的 L1 距离
        loss_x = self.criterion(g_x_x, g_y_x)
        loss_y = self.criterion(g_x_y, g_y_y)
        
        return loss_x + loss_y


class Pix2PixModel(BaseModel):
    """This class implements the pix2pix model, for learning a mapping from input images to output images given paired data."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm="batch", netG="unet_256", dataset_mode="aligned")
        if is_train:
            parser.set_defaults(pool_size=0)
            
            # --- 修改 1: 默认使用 lsgan 模式 (解决判别器过强导致梯度消失的问题) ---
            parser.set_defaults(gan_mode="lsgan") 
            
            # L1 Loss 权重
            parser.add_argument("--lambda_L1", type=float, default=100.0, help="weight for L1 loss")
            
            # --- 修改 2: 新增 GDL Loss 权重参数 (默认 100.0) ---
            parser.add_argument("--lambda_GDL", type=float, default=100.0, help="weight for gradient difference loss")

        return parser

    def get_current_metrics(self):
        """返回当前的 PSNR, SSIM 和 Masked PSNR 值"""
        metrics = self._calculate_metrics()
        
        if hasattr(self, 'real_B') and hasattr(self, 'fake_B'):
            masked_psnr_val = calculate_masked_psnr(self.fake_B, self.real_B)
            metrics['PSNR_Masked'] = masked_psnr_val
            
        return metrics

    def _calculate_metrics(self):
        """计算标准 PSNR 和 SSIM"""
        if not hasattr(self, 'real_B') or self.real_B.shape[0] == 0:
            return {}
            
        real_B_np = self.real_B[0].cpu().detach().numpy().transpose(1, 2, 0)
        fake_B_np = self.fake_B[0].cpu().detach().numpy().transpose(1, 2, 0)
        DATA_RANGE = 2.0
        
        psnr_value = psnr(real_B_np, fake_B_np, data_range=DATA_RANGE)
        
        if real_B_np.shape[-1] == 1:
            real_B_np = real_B_np.squeeze(-1)
            fake_B_np = fake_B_np.squeeze(-1)

        ssim_value = ssim(
            real_B_np,  
            fake_B_np,  
            data_range=DATA_RANGE,
            multichannel=False
        )
        
        return {'PSNR': float(f'{psnr_value:.4f}'), 'SSIM': float(f'{ssim_value:.4f}')}

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # --- 修改 3: loss_names 中加入 'G_GDL'，移除 'G_VGG' ---
        self.loss_names = ["G_GAN", "G_L1", "G_GDL", "D_real", "D_fake"]
        
        self.metric_names = ['PSNR', 'SSIM', 'PSNR_Masked'] 
            
        self.visual_names = ["real_A", "fake_B", "real_B"]
        if self.isTrain:
            self.model_names = ["G", "D"]
        else:
            self.model_names = ["G"]
        self.device = opt.device
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain)

        if self.isTrain:
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)

        if self.isTrain:
            # 这里的 gan_mode 已经被 set_defaults 修改为 'lsgan'
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            
            # --- 修改 4: 初始化梯度损失 ---
            self.criterionGDL = GradientLoss().to(self.device)

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        # 1. GAN Loss (LSGAN, 因为 gan_mode='lsgan')
        # 这会让判别器不仅仅是判别真假，而是拉近真假分布的距离，梯度更平滑
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        
        # 2. L1 Loss (保证像素准确性)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        
        # 3. --- 新增: Gradient Difference Loss (保证边缘锐利度) ---
        # 这里的 lambda_GDL 默认为 100，和 L1 保持 1:1 的比例通常效果最好
        self.loss_G_GDL = self.criterionGDL(self.fake_B, self.real_B) * self.opt.lambda_GDL
        
        # 4. 总 Loss
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_GDL
        
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
