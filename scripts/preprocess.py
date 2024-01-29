#!/usr/bin/env python

import cv2 
import os 
from tqdm import tqdm  

def read_image_rgb(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def split_image_and_mask(img_and_mask):
    height = img_and_mask.shape[0]
    img = img_and_mask[:, :height]
    mask = img_and_mask[:, height:]
    return img, mask

splits = ['train', 'val']
dataset = 'cityscapes_data'
save_dir = os.path.join(os.path.dirname(__file__), '../training_data')
img_size = (256, 256) 

for split in splits:
    data_dir = os.path.join(dataset, split)
    img_dir = os.path.join(save_dir, split, 'img')
    mask_dir = os.path.join(save_dir, split, 'mask')

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    paths  = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]

    # split image and mask
    for i, path in tqdm(enumerate(paths)):
        img_and_mask = read_image_rgb(path)
        img, mask = split_image_and_mask(img_and_mask)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = (
            cv2.resize(img, img_size)
            if img.shape[0] != img_size[1] and img.shape[1] != img_size[0]
            else img
        )
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
        mask = (
            cv2.resize(mask, img_size)
            if mask.shape[0] != img_size[1] and mask.shape[1] != img_size[0]
            else mask
        )
        file_name = os.path.basename(path)
        cv2.imwrite(os.path.join(img_dir, file_name), img)
        cv2.imwrite(os.path.join(mask_dir, file_name), mask)