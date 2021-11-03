from utils.emd_dot_trainer import EMDTrainer 
import argparse
import os
import torch
args = None

# d_set = './UCF-QNRF', '../nwpu_bayesloss_format'

def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--save-dir', default='./checkpoints/vggcsr2',
                        help='directory to save models.')
    parser.add_argument('--data-dir', default='/home/students/s121md106_04/GLoss_processed/NWPU-Train-Val-Test-offset-1',
                        help='training data directory')
    parser.add_argument('--o_cn', type=int, default=1,
                        help='output channel number')
    parser.add_argument('--cost', type=str, default='per',
                        help='cost distance')
    parser.add_argument('--scale', type=float, default=0.6,
                        help='scale for coordinates')
    parser.add_argument('--reach', type=float, default=0.5,
                        help='reach')
    parser.add_argument('--blur', type=float, default=0.01,
                        help='blur')
    parser.add_argument('--scaling', type=float, default=0.5,
                        help='scaling')
    parser.add_argument('--tau', type=float, default=0.1,
                        help='blur')
    parser.add_argument('--p', type=float, default=1,
                        help='p')
    parser.add_argument('--d_point', type=str, default='l1',
                        help='divergence for point loss')
    parser.add_argument('--d_pixel', type=str, default='l2',
                        help='divergence for pixel loss')

    # ../ucf_vgg19_ot_84.pth
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='the initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, ## default 1e-5
                        help='the weight decay')
    parser.add_argument('--resume', default='/home/students/s121md106_04/GeneralizedLoss-Counting-Pytorch-main/checkpoints/vggcsr2/1016-131052/499_ckpt.tar',
                        help='the path of resume training model')
    parser.add_argument('--max-model-num', type=int, default=1,
                        help='max models num to save ')
    parser.add_argument('--max-epoch', type=int, default=1000,
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=5,
                        help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=10,
                        help='the epoch start to val')

    parser.add_argument('--batch-size', type=int, default=1,
                        help='train batch size')
    parser.add_argument('--device', default='3', help='assign device')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='the num of training process')

    parser.add_argument('--is-gray', type=bool, default=False,
                        help='whether the input image is gray')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='the crop size of the train image')
    parser.add_argument('--downsample-ratio', type=int, default=8,
                        help='downsample ratio') # original val = 8

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.makedirs('./checkpoints/temp', exist_ok=True)
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = EMDTrainer(args)
    trainer.setup()
    trainer.train()
