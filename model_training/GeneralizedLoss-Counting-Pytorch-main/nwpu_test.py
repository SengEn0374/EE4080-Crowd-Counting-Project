import torch
import os
import numpy as np
from models.vgg19csr import vgg19
# from models.vgg19csr3 import vgg19
# from models.vgg import vgg19
import argparse
from PIL import Image
import torchvision.transforms as standard_transforms
import cv2
from matplotlib import pyplot as plt
from matplotlib import cm as c

mean_std = ([0.446139603853, 0.409515678883, 0.395083993673], [0.288205742836, 0.278144598007, 0.283502370119])
img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

# './UCF-QNRF/test'
def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--save-dir', default='/home/students/s121md106_04/GeneralizedLoss-Counting-Pytorch-main/checkpoints/vggcsr2/1019-005149/best_val_750ep_52.30mae_226.19mse.pth',
                        help='model path')
    parser.add_argument('--device', default='0', help='assign device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    dataRoot = '/home/students/s121md106_04/ProcessedDataset/NWPU-Crowd/min_576x768_mod16_2048'
    txtpath = os.path.join(dataRoot, 'txt_list', 'test.txt')
    with open(txtpath) as f:
        lines = f.readlines()

    # datasets = Crowd(args.data_dir, 512, 8, is_gray=False, method='val')
    # dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
    #                                          num_workers=1, pin_memory=False)

    model = vgg19()
    device = torch.device('cuda')
    model.to(device)
    checkpoint = torch.load(args.save_dir, device)
    model.load_state_dict(checkpoint)
    epoch_minus = []

    max = 0.0
    record = open('vggcsr2_submitted.txt', 'w+')
    for infos in lines:
        filename = infos.split()[0]
        name = filename
        imgname = os.path.join(dataRoot, 'img', filename + '.jpg')
        img = Image.open(imgname)
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
            pred = torch.sum(outputs).item()
            print(f'{filename} {pred:.4f}', file=record)
            print(f'{filename} {pred:.4f}')

        # # display overlay
        #
        #     # fetch density map
        #     temp = np.asarray(
        #         outputs.detach().cpu().reshape(outputs.detach().cpu().shape[2], outputs.detach().cpu().shape[3]))
        #     # print(temp.min()) # negative values causing wrap around
        #     temp = temp - temp.min()  # shift to zero base value
        #     if temp.max() > max:
        #         max = temp.max()
        #         print("max", max)
        #     frame = np.asarray(img)
        #     temp = cv2.resize(temp, (frame.shape[1], frame.shape[0]))
        #
        #     # scale back to img size + convert to unint8 for map reading as image
        #     scale = 255 / temp.max()  # want to set fixed color mapping instead?
        #     # scale = 255/ density_norm_factor
        #
        #     temp = (temp * scale).astype(np.uint8)
        #     temp = cv2.applyColorMap(temp, cv2.COLORMAP_JET)
        #
        #     # convert frame back to rgb (cv2 read as bgr)
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #
        #     # overlay frame with density map
        #     overlay = cv2.addWeighted(frame, 0.5, temp, 0.5, 0.0)
        #
        #     # add count number on screen
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     cv2.putText(overlay, str(pred), (50, 50), font, 1, (225, 225, 225), 3)
        #
        #     cv2.imshow('overlay', overlay)
        #     cv2.waitKey(0)
        #     # cv2.imshow('temp', temp)
        #     # k = cv2.waitKey(1) & 0xff
        #     # if k == 27:  # press 'ESC' to quit
        #     #     break
        #
        # cv2.destroyAllWindows()
