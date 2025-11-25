"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pretrained models):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a pix2pix model (direction BtoA):
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch  # 确保导入 torch

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    
    # >>>>>>>>>>>> 修复逻辑开始：自动检测 GPU 并设置 device <<<<<<<<<<<<
    # 1. 如果 opt 中没有 gpu_ids 属性，手动根据硬件情况创建一个
    if not hasattr(opt, 'gpu_ids'):
        opt.gpu_ids = [0] if torch.cuda.is_available() else []
    
    # 2. 根据 gpu_ids 设置 device
    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        opt.device = torch.device('cuda:{}'.format(opt.gpu_ids[0]))
    else:
        opt.device = torch.device('cpu')
    # >>>>>>>>>>>>>>>>>>>> 修复逻辑结束 <<<<<<<<<<<<<<<<<<<<

    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test script saves the results to a HTML file.
    
    # --- 确保不计算梯度，节省显存 ---
    torch.no_grad()

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{}_iter{}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    
    # --- 初始化指标统计 ---
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    
    if opt.eval:
        model.eval()
        
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
            
        model.set_input(data)  # unpack data from dataset and apply preprocessing
        model.test()           # run inference
        
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        
        if i % 5 == 0:  # print every 5 images
            print('processing (%04d)-th image... %s' % (i, img_path))

        # --- 新增: 计算并累加 PSNR/SSIM ---
        if hasattr(model, 'get_current_metrics'):
            metrics = model.get_current_metrics()
            # print(f"  Image {i}: PSNR={metrics['PSNR']:.4f}, SSIM={metrics['SSIM']:.4f}") # 可选打印
            
            total_psnr += metrics['PSNR']
            total_ssim += metrics['SSIM']
            count += 1
        # --------------------------------

        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    
    webpage.save()  # save the HTML

    # --- 新增: 打印平均指标 ---
    if count > 0:
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        print("\n" + "="*50)
        print(f" Test Set Evaluation Results ({count} images)")
        print(f" Average PSNR: {avg_psnr:.4f}")
        print(f" Average SSIM: {avg_ssim:.4f}")
        print("="*50 + "\n")
        
        # 可选：将结果保存到文本文件
        with open(os.path.join(web_dir, 'test_metrics.txt'), 'w') as f:
            f.write(f"Average PSNR: {avg_psnr:.4f}\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")
    else:
        print("No metrics calculated (maybe get_current_metrics is missing or no paired data).")
