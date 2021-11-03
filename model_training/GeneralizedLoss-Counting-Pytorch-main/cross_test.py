'''
V3 uses generalized loss preprocessed datasets instead of nwpu format in v2 and v1
'''
import torch
import numpy as np
import os
import argparse
# from models.vgg import vgg19
# from models.vgg16 import vgg16
# from models.vgg19csr3 import vgg19
from models.vgg19csr import vgg19
from PIL import Image
import torchvision.transforms as standard_transforms

d_sets_dirs = [('/home/students/s121md106_04/GLoss_processed/SHT_B-Train-Val-Test/test', 'SHT-B'),
               ('/home/students/s121md106_04/GLoss_processed/SHT_A-Train-Val-Test/test', 'SHT-A'),
               ('/home/students/s121md106_04/GLoss_processed/UCF-QNRF-bayesloss/test', 'UCF'),
               ('/home/students/s121md106_04/GLoss_processed/JHU-Train-Val-Test/test', 'JHU'),
               ('/home/students/s121md106_04/GLoss_processed/NWPU-Train-Val-Test-offset-1/val', 'NWPU')]

# d_sets_dirs = [('/home/students/s121md106_04/GLoss_processed/UCF-QNRF-bayesloss/test', 'UCF')]

weights_dir = '/home/students/s121md106_04/GeneralizedLoss-Counting-Pytorch-main/checkpoints/vggcsr2/1019-005149/best_val_750ep_52.30mae_226.19mse.pth'
print('vggcsr2, < 1000ep')
# '/home/students/s121md106_04/GeneralizedLoss-Counting-Pytorch-main/checkpoints/vgg19csr2_GLoss_nwpu/0916-195251/best_val_ep924.pth' 
#'/home/students/s121md106_04/GeneralizedLoss-Counting-Pytorch-main/checkpoints/temp/0902-105136/best_val.pth'
#'/home/students/s121md106_04/GeneralizedLoss-Counting-Pytorch-main/checkpoints/stock_nwpu_1e-6/0929-000800/best_val.pth'
#'/home/students/s121md106_04/GeneralizedLoss-Counting-Pytorch-main/checkpoints/stock_nwpu/0928-235731/best_val.pth'
# '/home/students/s121md106_04/ucf_vgg19_ot_84.pth'

mean_std_nwpu = ([0.446139603853, 0.409515678883, 0.395083993673], [0.288205742836, 0.278144598007, 0.283502370119])
mean_std_imgnet = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std_nwpu)])

# device = torch.device('cuda')


def main():
    model = vgg19()
    model.load_state_dict(torch.load(args.weight_dir))

    print(f"using gpu {args.device} for testing.")
    device = torch.device(args.device)
    model.to(device).eval()


    for dset in d_sets_dirs:
        imgs = os.listdir(dset[0])
        cumulative_error = 0
        cumulative_sqr_error = 0

        epoch_minus = []
        j = 0
        for i, img in enumerate(imgs):
            if '.npy' in img:
                continue
            # if '1329' in img: # val
            #     continue
            # if i<4782:
            #     continue
            img_path = os.path.join(dset[0], img)

            gt_path = img_path.replace('.jpg', '.npy')
            gt = np.load(gt_path)
            count = len(gt)

            im = Image.open(img_path)
            if im.mode == 'L':
                im = im.convert('RGB')
            inputs = img_transform(im)[None, :, :, :].to(device)

            # print(i, img)
            with torch.no_grad():
                output = model(inputs)
                pred = torch.sum(output).item()

            temp_minu = count - torch.sum(output).item()
            # print(img, temp_minu, count, torch.sum(output).item())
            epoch_minus.append(temp_minu)

        epoch_minus = np.array(epoch_minus)
        mse = np.sqrt(np.mean(np.square(epoch_minus)))
        mae = np.mean(np.abs(epoch_minus))
        log_str = '{} Test: mae {}, mse {}'.format(dset[1], mae, mse)
        print(log_str)
        j+=1

        #     error = abs(pred-count)
        #     sqred_err = error**2
        #     cumulative_error += error
        #     cumulative_sqr_error += sqred_err

        # num_imgs = len(imgs)
        # mae = cumulative_error / num_imgs
        # mse = cumulative_sqr_error / num_imgs
        # log_str = '{} Test: mae {}, mse {}'.format(dset[1], mae, mse)
        # print(log_str)

        




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data_dirs', default=d_sets_dirs, help='testing dataset directories list')
    parser.add_argument('--weight_dir', default=weights_dir, help='model path')
    parser.add_argument('--device', default=3, help='assign device')
    args = parser.parse_args()

    main()
