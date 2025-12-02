import torch
from .base_model import BaseModel
from . import networks
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import numpy as np
# --- 新增: 导入 Masked PSNR 计算函数 ---
from util.util import calculate_masked_psnr


class Pix2PixModel(BaseModel):
    """This class implements the pix2pix model, for learning a mapping from input images to output images given paired data."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm="batch", netG="unet_256", dataset_mode="aligned")
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode="vanilla")
            parser.add_argument("--lambda_L1", type=float, default=100.0, help="weight for L1 loss")

        return parser

    def get_current_metrics(self):
        """返回当前的 PSNR, SSIM 和 Masked PSNR 值"""
        # 1. 计算原有的全图指标 (使用 skimage)
        metrics = self._calculate_metrics()
        
        # 2. 计算 Masked PSNR (使用 PyTorch)
        if hasattr(self, 'real_B') and hasattr(self, 'fake_B'):
            masked_psnr_val = calculate_masked_psnr(self.fake_B, self.real_B)
            metrics['PSNR_Masked'] = masked_psnr_val
            
        return metrics

    def _calculate_metrics(self):
        """计算标准 PSNR 和 SSIM (全图计算)"""
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
        self.loss_names = ["G_GAN", "G_L1", "D_real", "D_fake"]
        
        # --- 修改: 添加 'PSNR_Masked' 到指标列表 ---
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
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
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
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
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
