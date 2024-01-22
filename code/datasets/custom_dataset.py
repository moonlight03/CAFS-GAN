"""
TUNIT: Truly Unsupervised Image-to-Image Translation
Copyright (c) 2020-present NAVER Corp.
MIT license 
"""
import torch.utils.data as data

from PIL import Image
import random
import os
import os.path
import sys
import torch
import copy

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, mode, content_path, imagenums):
    images = []
    image_path = sorted(os.listdir(dir))
    text_effect, glyph = set(), set()

    hash_images = {}

    for i in image_path:
        if i.__contains__('-'):
            gl, te = i.split('-')
            glyph.add(gl)
            text_effect.add(te)

    text_effect, glyph = sorted(list(text_effect)), sorted(list(glyph))
    labels, label_list = [], []
    for i in image_path:
        if i.__contains__('-'):
            gl, te = i.split('-')
            index = (glyph.index(gl), text_effect.index(te))
            label_list.append(index)
            t = os.path.join(dir, i)
            imgs1 = []
            if mode == 'train':
                imgs1 += sorted([os.path.join(t, j) for j in os.listdir(t)])[: imagenums]
                random.shuffle(imgs1)
            else:
                imgs1 += sorted([os.path.join(t, j) for j in os.listdir(t)])[: imagenums]

            labels += [(glyph.index(gl), text_effect.index(te)) for j in os.listdir(t)][: imagenums]

            hash_images[index] = imgs1
            images += imgs1

    glyph_list = []
    effect_list = []
    for i in range(len(glyph)):
        tmp = []
        for j in label_list:
            if j[0] == i:
              tmp.append(j)
        glyph_list.append(tmp)
    for i in range(len(text_effect)):
        tmp = []
        for j in label_list:
            if j[1] == i:
                tmp.append(j)
        effect_list.append(tmp)

    if mode == 'train':
        random.shuffle(images)  # 这个images是当做content来用的
        return images, hash_images, labels, glyph_list, effect_list
    else:
        source_path = os.path.join(dir, content_path)
        source_list = sorted([os.path.join(source_path, p) for p in os.listdir(source_path)])[: imagenums]
        return source_list, images, hash_images, labels, glyph_list, effect_list  # 0-8, 0-8, ...





class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions, transform=None, target_transform=None, mode='train', content_path='', imagenums=775):
        if mode == 'train':
            images, hash_images, labels, glyph, text_effect = make_dataset(root, mode, content_path, imagenums)
        else:
            source, images, hash_images, labels, glyph, text_effect = make_dataset(root, mode, content_path, imagenums)
            self.source = source
        self.root = root
        self.loader = loader
        self.extensions = extensions
        self.len_g = len(glyph)
        self.len_t = len(text_effect)
        self.glyph_list = glyph
        self.effect_list = text_effect
        self.samples = images
        self.hash_images = hash_images
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        imgname = path.split('/')[-1].replace('.JPEG', '')
        return sample, target, imgname

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolerRemap(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, remap_table=None, with_idx=False, mode='train', content_path='', imagenums=972):
        super(ImageFolerRemap, self).__init__(root, loader, IMG_EXTENSIONS, transform=transform, target_transform=target_transform, mode=mode, content_path=content_path, imagenums=imagenums)

          # 三类的所有图像
        self.mode = mode
        self.targets_final = [a * self.len_t + b for a, b in self.labels]
        self.class_table = remap_table
        self.with_idx = with_idx

    def __getitem__(self, index):

        if self.mode == 'train':

            target = random.sample(self.labels, 1)[0]
            Xs_label = copy.copy(self.glyph_list[target[0]])
            X = random.sample(Xs_label, 1)[0]
            Xs_label.remove(X)
            X_aug_label = random.sample(Xs_label, 1)[0]
            X_aug_paths = random.sample(self.hash_images[X_aug_label], 1)

            Ys_label = copy.copy(self.effect_list[target[1]])
            Y = random.sample(Ys_label, 1)[0]
            Ys_label.remove(Y)
            Y_aug_label = random.sample(Ys_label, 1)[0]
            Y_aug_paths = random.sample(self.hash_images[Y_aug_label], 1)

            path1 = random.sample(self.hash_images[X], 1)[0]
            path2 = random.sample(self.hash_images[Y], 1)[0]
            source_img = random.sample(self.samples, 1)[0]
            sample1 = self.loader(path1)
            sample1 = self.transform(sample1)
            sample2 = self.loader(path2)
            sample2 = self.transform(sample2)
            source_img = self.loader(source_img)
            source_img = self.transform(source_img)

            X_aug = self.transform(self.loader(X_aug_paths[0]))
            Y_aug = self.transform(self.loader(Y_aug_paths[0]))

            glyph_list_neg = copy.copy(self.glyph_list)
            glyph_list_neg = [glyph_list_neg[i] for i in range(len(glyph_list_neg)) if i != target[0]]
            effect_list_neg = copy.copy(self.effect_list)
            effect_list_neg = [effect_list_neg[i] for i in range(len(effect_list_neg)) if i != target[1]]
            X_negs = torch.randn([len(glyph_list_neg), sample1.size(0), sample1.size(1), sample1.size(2)])
            Y_negs = torch.randn([len(effect_list_neg), sample1.size(0), sample1.size(1), sample1.size(2)])
            for i in range(len(glyph_list_neg)):
                X_negs[i] = self.transform(self.loader(random.sample(self.hash_images[random.sample(glyph_list_neg[i], 1)[0]], 1)[0]))
            for i in range(len(effect_list_neg)):
                Y_negs[i] = self.transform(self.loader(random.sample(self.hash_images[random.sample(effect_list_neg[i], 1)[0]], 1)[0]))

            return sample1, sample2, torch.tensor(X[0]), torch.tensor(Y[1]), source_img, X_aug, Y_aug, X_negs, Y_negs
        else:

            img = self.samples[index]
            img = self.loader(img)
            img = self.transform(img)
            source_img = torch.rand_like(img)
            if index < len(self.source):
                source_img = self.source[index]
                source_img = self.loader(source_img)
                source_img = self.transform(source_img)
            return img, source_img

    def __len__(self):
        if self.mode == 'train':
            return 99999999
        else:
            return len(self.samples)


