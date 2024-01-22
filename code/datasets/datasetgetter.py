"""
TUNIT: Truly Unsupervised Image-to-Image Translation
Copyright (c) 2020-present NAVER Corp.
MIT license 
"""
import torch
from torchvision.datasets import ImageFolder
import os
import torchvision.transforms as transforms
from datasets.custom_dataset import ImageFolerRemap


class DuplicatedCompose(object):
    def __init__(self, tf1, tf2):
        self.tf1 = tf1
        self.tf2 = tf2

    def __call__(self, img):
        img1 = img.copy()
        img2 = img.copy()
        for t1 in self.tf1:
            img1 = t1(img1)
        for t2 in self.tf2:
            img2 = t2(img2)
        return img1, img2


def get_dataset(dataset, args):

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    normalize = transforms.Normalize(mean=mean, std=std)

    transform = transforms.Compose([

                                   transforms.ToTensor(),
                                   normalize])

    transform_val = transforms.Compose([

                                       transforms.ToTensor(),
                                       normalize])


    class_to_use = args.att_to_use

    print('USE CLASSES', class_to_use)

    remap_table = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    print("LABEL MAP:", remap_table)
    print('USE Animals ImageNet Subset dataset [WITH IIC]')
    print("LABEL MAP:", remap_table)

    img_dir_train = os.path.join(args.data_dir, args.train_dataset)
    img_dir_test = os.path.join(args.data_dir, args.test_dataset)  # use for zero-shot


    dataset = ImageFolerRemap(img_dir_train, transform=transform, remap_table=remap_table, mode='train', content_path=args.content_path, imagenums=args.train_imagenums)
    valdataset = ImageFolerRemap(img_dir_test, transform=transform_val, remap_table=remap_table, mode='test', content_path=args.content_path, imagenums=args.test_contentimagenums)
    # parse classes to use
    return dataset, valdataset