import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class FIDDataset(Dataset):

    def __init__(self, path, mode=None, test_imagenums=972):

        if mode is None:
            D_path = os.path.join(path)
            list_D = sorted([os.path.join(D_path, i) for i in os.listdir(D_path)])[:test_imagenums]
            self.list_D = list_D
        elif mode == 'mean':
            D_path = os.path.join(path)
            list_D = sorted([os.path.join(D_path, i) for i in os.listdir(D_path) if i.__contains__('mean')])[:test_imagenums]
            self.list_D = list_D
        else:
            D_path = os.path.join(path)
            list_D = sorted([os.path.join(D_path, i) for i in os.listdir(D_path) if not i.__contains__('mean')])[:test_imagenums]
            self.list_D = list_D


    def __getitem__(self, index):
        D_path = self.list_D[index]
        img_d = Image.open(D_path)
        d = self.get_transform_T()(img_d)
        return d

    def __len__(self):
        return len(self.list_D)

    def get_transform_T(self, method=Image.BICUBIC, convert=True):
        transform_list = []
        if convert:
            transform_list += [transforms.Resize([128,128]), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)
