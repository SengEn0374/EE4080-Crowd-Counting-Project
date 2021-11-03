import torch
import os
import numpy as np
from models.vgg import vgg19
# from models.vgg19csr import vgg19
# from models.vgg19csr3 import vgg19
import argparse
from PIL import Image, ImageOps
import torchvision.transforms as standard_transforms
import cv2
from matplotlib import pyplot as plt
from matplotlib import cm as c

mean_std_nwpu = ([0.446139603853, 0.409515678883, 0.395083993673], [0.288205742836, 0.278144598007, 0.283502370119])
mean_std_imgnet = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std_nwpu)
    ])

# './UCF-QNRF/test'
# vggcsr2 ucf '/home/students/s121md106_04/GeneralizedLoss-Counting-Pytorch-main/checkpoints/vgg19csr2_GLoss_ucf_ImageNetMeanStdTransform/0922-233106/best_val.pth', # vggcsr2 ucf 1k ep
# /home/students/s121md106_04/GeneralizedLoss-Counting-Pytorch-main/checkpoints/vggcsr3/1010-123945/best_val_ep710_48mae_123mse.pth
# '/home/students/s121md106_04/GeneralizedLoss-Counting-Pytorch-main/checkpoints/vggcsr2/1019-005149/best_val_750ep_52.30mae_226.19mse.pth'
def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--save-dir', default='/home/students/s121md106_04/GeneralizedLoss-Counting-Pytorch-main/checkpoints/stock_nwpu/0928-235731/best_val.pth',
                        help='model path')
    parser.add_argument('--device', default='1', help='assign device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    # dataRoot = '../NWPU-Crowd-Sample-Code/ProcessedDataset/NWPU-Crowd/min_576x768_mod16_2048'
    # txtpath = os.path.join(dataRoot, 'txt_list', 'test.txt')
    # with open(txtpath) as f:
    #     lines = f.readlines()
    # datasets = Crowd(args.data_dir, 512, 8, is_gray=False, method='val')
    # dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
    #                                          num_workers=1, pin_memory=False)

    data_path = '/home/students/s121md106_04/temp'
    # data_path = '/home/students/s121md106_04/test_data_final'
    lines = os.listdir(data_path)

    save_dir = '/home/students/s121md106_04/GeneralizedLoss-Counting-Pytorch-main/inference/output_stock_nwpu_1510'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = vgg19()
    device = torch.device('cuda')
    model.to(device)
    checkpoint = torch.load(args.save_dir, device)
    model.load_state_dict(checkpoint)
    epoch_minus = []

    max = 0.0
    # record = open('submmited.txt', 'w+')
    # for infos in lines:
    #     filename = infos.split()[0]
    #     name = filename
    #     imgname = os.path.join(dataRoot, 'img', filename + '.jpg')
    for filename in lines:
        name = filename.split('.')[0]
        imgname = os.path.join(data_path, filename)
        img = Image.open(imgname)
        img = ImageOps.exif_transpose(img)
        if img.mode == 'L':
            img = img.convert('RGB')
        inputs = img_transform(img)[None, :, :, :]
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            # plt.imshow(outputs[0,0,:,:].cpu().numpy().astype(int), cmap=c.jet)
            # plt.show()
            # print(outputs.shape)
            pred = int(torch.sum(outputs).item())
            # print(f'{filename} {pred:.4f}', file=record)
            print(f'{name} {pred:.4f}')

        # display overlay

            # fetch density map
            temp = np.asarray(
                outputs.detach().cpu().reshape(outputs.detach().cpu().shape[2], outputs.detach().cpu().shape[3]))
            # print(temp.min()) # negative values causing wrap around
            temp = temp - temp.min()  # shift to zero base value
            if temp.max() > max:
                max = temp.max()
                # print("max", max)
            frame = np.asarray(img)
            temp = cv2.resize(temp, (frame.shape[1], frame.shape[0]))

            if temp.max() == 0:
                scale = 1
            else: 
                # scale back to img size + convert to unint8 for map reading as image
                scale = 255 / temp.max()  # want to set fixed color mapping instead?
                # scale = 255/ density_norm_factor

            temp = (temp * scale).astype(np.uint8)
            temp = cv2.applyColorMap(temp, cv2.COLORMAP_JET)

            # convert frame back to rgb (cv2 read as bgr)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # overlay frame with density map
            overlay = cv2.addWeighted(frame, 0.5, temp, 0.5, 0.0)

            # add count number on screen
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(overlay, str(pred), (50, 50), font, 1, (225, 225, 225), 3)

            # cv2.imshow('overlay', overlay)
            cv2.imwrite(save_dir+f'/{name}.jpg', overlay)
            cv2.waitKey(0)
            # cv2.imshow('temp', temp)
            # k = cv2.waitKey(1) & 0xff
            # if k == 27:  # press 'ESC' to quit
            #     break

        cv2.destroyAllWindows()

'''
import torch
import os
import numpy as np
from datasets.crowd import Crowd
from models.vgg import vgg19
import argparse
from matplotlib import pyplot as plt
from matplotlib import cm as c
import cv2

# './UCF-QNRF/test'
def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default='C:/Users/crono/Desktop/NWPU-Crowd-Sample-Code/ProcessedDataset/UCF-QNRF_Val_NWPU_FORMAT/img',
                        help='test data directory')
    parser.add_argument('--save-dir', default='C:/Users/crono/Desktop/ucf_vgg19_ot_84.pth',
                        help='model path')
    parser.add_argument('--device', default='0', help='assign device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    # datasets = Crowd(os.path.join(args.data_dir, 'test'), 512, 8, is_gray=False, method='val')
    datasets = Crowd(args.data_dir, 512, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=1, pin_memory=False)
    model = vgg19()
    device = torch.device('cuda')
    model.to(device)
    checkpoint = torch.load(args.save_dir)
    model.load_state_dict(checkpoint)
    epoch_minus = []

    for inputs, count, name in dataloader:
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            # plt.imshow(outputs[0,0,:,:].cpu().numpy().astype(int), cmap=c.jet)
            # plt.show()
            # print(outputs.shape)
            temp_minu = count.numpy()[0] - torch.sum(outputs).item()
            print(name, temp_minu, count.numpy()[0], torch.sum(outputs).item())
            epoch_minus.append(temp_minu)

    epoch_minus = np.array(epoch_minus)
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    mae = np.mean(np.abs(epoch_minus))
    log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
    print(log_str)
'''