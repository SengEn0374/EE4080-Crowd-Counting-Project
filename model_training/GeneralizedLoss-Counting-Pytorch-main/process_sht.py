'''
moodify 'A' to 'B' in data_dir and save_dir to process 'part B'
'''

from scipy.io import loadmat
from PIL import Image
import numpy as np
import os
import cv2
import argparse

def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w*ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w*ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h*ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h*ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio


def find_dis(point):
    square = np.sum(point*points, axis=1)
    dis = np.sqrt(np.maximum(square[:, None] - 2*np.matmul(point, point.T) + square[None, :], 0.0))
    dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
    return dis

def generate_data(im_path):
    im = Image.open(im_path)
    im_w, im_h = im.size
    mat_path = im_path.replace('images', 'ground_truth').replace('IMG', 'GT_IMG').replace('.jpg', '.mat')
    points = loadmat(mat_path)['image_info'][0,0][0,0][0].astype(np.float32)
    # points = loadmat(mat_path)['annPoints'].astype(np.float32)
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr
    return Image.fromarray(im), points


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--origin-dir', default='F:/EE4080 - Final Year Project/Datasets/unprocessed/ShanghaiTech_Crowd_Counting_Dataset/part_A_final', help='original data directory')
    parser.add_argument('--data-dir', default='../SHT_A-Train-Val-Test', help='processed data directory')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    save_dir = args.data_dir
    min_size = 512
    max_size = 2048

    for phase in ['train_data', 'test_data']:
        sub_save_dir = os.path.join(save_dir, phase.strip('_data'))
        if not os.path.exists(sub_save_dir):
            os.makedirs(sub_save_dir)
        sub_data_dir = os.path.join(args.origin_dir, phase, 'images')
        img_list = os.listdir(sub_data_dir)
        for img in img_list:
            print(img)
            im_path = os.path.join(sub_data_dir, img)
            im, points = generate_data(im_path)
            if phase == 'train':
                if len(points) != 0:
                    dis = find_dis(points)
                    # print(points, dis)
                    points = np.concatenate((points, dis), axis=1)
            im_save_path = os.path.join(sub_save_dir, img)
            im.save(im_save_path)
            gd_save_path = im_save_path.replace('jpg', 'npy')
            np.save(gd_save_path, points)


