"""
TUNIT: Truly Unsupervised Image-to-Image Translation
Copyright (c) 2020-present NAVER Corp.
MIT license 
"""

import math

import cv2

import torch.utils.data
import numpy as np
from scipy import linalg
from validation.inception1 import inception_v3
from PIL import Image
from validation.fiddataset import FIDDataset
import torchvision.transforms as transforms
import torch.nn
from torch import nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils as vutils
import random
from tools.utils import *
from validation.vgg19 import Vgg19
from torch.autograd import Variable
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x):
        return x

IMAGENET_MEAN_255 = [0.5, 0.5, 0.5]
IMAGENET_STD_NEUTRAL = [0.5, 0.5, 0.5]
def prepare_model(device):
    '''
    Load VGG19 model into local cache.
    '''
    model = Vgg19(requires_grad=False, show_progress=True)
    content_feature_maps_index = model.content_feature_maps_index
    style_feature_maps_indices = model.style_feature_maps_indices
    layer_names = model.layer_names
    content_fms_index_name = (content_feature_maps_index, layer_names[content_feature_maps_index])
    style_fms_indices_names = (style_feature_maps_indices, layer_names)
    return model.to(device).eval(), content_fms_index_name, style_fms_indices_names

def load_image(img_path,target_shape="None"):
    '''
    Load and resize the image.
    '''
    if not os.path.exists(img_path):
        raise Exception(f'Path not found: {img_path}')
    img = cv2.imread(img_path)[:, :, ::-1]                   # convert BGR to RGB when reading
    # if target_shape is not None:
    #     if isinstance(target_shape, int) and target_shape != -1:
    #         current_height, current_width = img.shape[:2]
    #         new_height = target_shape
    #         new_width = int(current_width * (new_height / current_height))
    #         img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
    #     else:
    #         img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)
    img = img.astype(np.float32)
    img /= 255.0
    return img

def prepare_img(img_path, target_shape, device):
    '''
    Normalize the image.
    '''
    img = load_image(img_path, target_shape=target_shape)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([128, 128]),
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)])
    img = transform(img).to(device).unsqueeze(0)
    return img

def gram_matrix(x, should_normalize=True):
    '''
    Generate gram matrices of the representations of content and style images.
    '''
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= ch * h * w
    return gram
def compute_FID(img1, img2, batch_size=1):
    device = torch.device("cuda:0")  # you can change the index of cuda

    N1 = len(img1)
    N2 = len(img2)
    n_act = 2048  # the number of final layer's dimension

    # Set up dataloaders
    dataloader1 = torch.utils.data.DataLoader(img1, batch_size=batch_size)
    dataloader2 = torch.utils.data.DataLoader(img2, batch_size=batch_size)


    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()

    # get the activations
    def get_activations(x):
        x = inception_model(x)[0]
        return x.cpu().data.numpy().reshape(batch_size, -1)

    act1 = np.zeros((N1, n_act))
    act2 = np.zeros((N2, n_act))

    data = [dataloader1, dataloader2]
    act = [act1, act2]
    for n, loader in enumerate(data):
        for i, batch in enumerate(loader, 0):
            batch = batch.to(device)
            batch_size_i = batch.size()[0]
            activation = get_activations(batch)

            act[n][i * batch_size:i * batch_size + batch_size_i] = activation

    # compute the activation's statistics: mean and std
    def compute_act_mean_std(act):
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma
    mu_act1, sigma_act1 = compute_act_mean_std(act1)
    mu_act2, sigma_act2 = compute_act_mean_std(act2)

    # compute FID
    def _compute_FID(mu1, mu2, sigma1, sigma2, eps=1e-6):
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        diff = mu1 - mu2

        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        FID = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

        return FID

    FID = _compute_FID(mu_act1, mu_act2, sigma_act1, sigma_act2)
    return FID
def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))
def calculate_rgb_psnr(img1, img2):
    """calculate psnr among rgb channel, img1 and img2 have range [0, 255]
    """
    n_channels = np.ndim(img1)
    sum_psnr = 0
    for i in range(n_channels):
        this_psnr = calculate_psnr(img1[:, :, i], img2[:, :, i])
        sum_psnr += this_psnr
    return sum_psnr / n_channels
def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(img1.shape[2]):
                ssims.append(ssim(img1[..., i], img2[..., i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)
def get_transform():
    transforms_ = [
        transforms.Resize([128,128]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    return transforms.Compose(transforms_)
def print_SSIM_PNSR(class_path, args):
    folder_GT = os.path.join(args.data_dir, args.test_dataset, args.target_path)
    folder_Gen = class_path
    l1_loss_file = os.path.join(args.res_dir, "loss.txt")


    crop_border = 4  # same with scale
    suffix = ''  # suffix for Gen images
    test_Y = False  # True: test Y channel only; False: test RGB channels

    PSNR_all = []
    SSIM_all = []
    img_list = sorted([os.path.join(folder_GT, i) for i in os.listdir(folder_GT) if not i.__contains__('mean')])[:]
    img_list1 = sorted([os.path.join(folder_Gen, i) for i in os.listdir(folder_Gen)])[:]

    if test_Y:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')

    for i, img_path in enumerate(img_list):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        # print(base_name)
        im_GT = cv2.imread(img_path)

        im_GT = cv2.resize(im_GT, (128, 128)) / 255.
        # 不同格式图像
        # im_Gen = cv2.imread(os.path.join(folder_Gen, base_name + suffix + '.tif')) / 255.

        im_Gen = cv2.imread(os.path.join(img_list1[i]))
        im_Gen = cv2.resize(im_Gen, (128, 128)) / 255.

        if test_Y and im_GT.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
            im_GT_in = bgr2ycbcr(im_GT)
            im_Gen_in = bgr2ycbcr(im_Gen)
        else:
            im_GT_in = im_GT
            im_Gen_in = im_Gen

        # crop borders
        if crop_border == 0:
            cropped_GT = im_GT_in
            cropped_Gen = im_Gen_in
        else:
            if im_GT_in.ndim == 3:
                cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border, :]
                cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border, :]
            elif im_GT_in.ndim == 2:
                cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border]
                cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border]
            else:
                raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im_GT_in.ndim))

        # 不同通道数（Y通道和RGB三个通道），需要更改
        # calculate PSNR and SSIM
        # PSNR = calculate_psnr(cropped_GT * 255, cropped_Gen * 255)
        PSNR = calculate_rgb_psnr(cropped_GT * 255, cropped_Gen * 255)

        SSIM = calculate_ssim(cropped_GT * 255, cropped_Gen * 255)
        # print('{:3d} - {:25}. \tPSNR: {:.4f} dB, \tSSIM: {:.4f}'.format(
        #     i + 1, base_name, PSNR, SSIM))
        PSNR_all.append(PSNR)
        SSIM_all.append(SSIM)
    Mean_format = 'Mean_PSNR: {:.4f}, Mean_SSIM: {:.4f}'
    Mean_PSNR = sum(PSNR_all) / len(PSNR_all)
    Mean_SSIM = sum(SSIM_all) / len(SSIM_all)
    with open(l1_loss_file, "a") as f:
        f.write(Mean_format.format(Mean_PSNR, Mean_SSIM) + '\n')


def print_L1(class_path, args, epoch):
    l1_loss_file = os.path.join(args.res_dir, "loss.txt")
    pathA = class_path
    pathGT = os.path.join(args.data_dir, args.test_dataset, args.target_path)
    pathLA = sorted([os.path.join(pathA, i) for i in os.listdir(pathA) if not i.__contains__('mean')])[:]
    pathLA_mean = sorted([os.path.join(pathA, i) for i in os.listdir(pathA) if i.__contains__('mean')])[:]
    pathLB = sorted([os.path.join(pathGT, i) for i in os.listdir(pathGT)])[:]
    sum_one_L1 = 0.0
    sum_one_L2 = 0.0
    sum_mean_L1 = 0.0
    sum_mean_L2 = 0.0
    i = 0
    while i < len(pathLB):
        imga = Image.open(pathLA[i])
        imga_mean = Image.open(pathLA_mean[i])
        imgb = Image.open(pathLB[i])
        a = get_transform()(imga)
        a_mean = get_transform()(imga_mean)
        b = get_transform()(imgb)

        sum_mean_L1 += torch.nn.L1Loss()(a_mean, b)
        sum_mean_L2 += torch.nn.MSELoss()(a_mean, b)
        sum_one_L1 += torch.nn.L1Loss()(a, b)
        sum_one_L2 += torch.nn.MSELoss()(a, b)
        i += 1
    with open(l1_loss_file, "a") as f:
        f.write('EPOCH :' + str(epoch) + '----------------------------------------------------------------------\n')
        f.write('L1 loss:' + "\t" + str(sum_one_L1 / len(pathLB)) + "\t" + str(sum_mean_L1 / len(pathLB)) + "\n")
        f.write('MSE loss:' + "\t" + str(sum_one_L2 / len(pathLB)) + "\t" + str(sum_mean_L2 / len(pathLB)) + "\n")

def nn_conv2d(im):
    # 用nn.Conv2d定义卷积操作
    conv_op = nn.Conv2d(1, 1, 3, bias=False)
    # 定义sobel算子参数
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    # 将sobel算子转换为适配卷积操作的卷积核
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    # 给卷积操作的卷积核赋值
    conv_op.weight.data = torch.from_numpy(sobel_kernel)
    # 对图像进行卷积操作
    edge_detect = conv_op(Variable(im))
    # 将输出转换为图片格式
    edge_detect = edge_detect.squeeze().detach().numpy()
    return edge_detect


def image2gray_ndarray(path):
    im = Image.open(path).resize((128, 128), Image.ANTIALIAS).convert('L')
    # 将图片数据转换为矩阵
    im = np.array(im, dtype='float32')
    # 将图片矩阵转换为pytorch tensor,并适配卷积输入的要求
    im = torch.from_numpy(im.reshape((1, 1, im.shape[0], im.shape[1])))
    # 边缘检测操作
    edge_detect = nn_conv2d(im)
    return edge_detect


def print_EdgeLoss(class_path, args):
    l1_loss_file = os.path.join(args.res_dir, "loss.txt")
    RE_Path = class_path
    GT_Path = os.path.join(args.data_dir, args.test_dataset, args.target_path)
    edge_loss = 0.0
    image_re_paths = [os.path.join(RE_Path, i) for i in os.listdir(RE_Path) if not i.__contains__('mean')]
    image_gt_paths = [os.path.join(GT_Path, i) for i in os.listdir(GT_Path)]
    for i in range(len(image_re_paths)):
        image_re = image2gray_ndarray(image_re_paths[i])
        image_gt = image2gray_ndarray(image_gt_paths[i])
        M = abs(image_re - image_gt)
        edge_loss += np.sum(np.reshape(M, (M.size,))) / (len(image_re) * len(image_re))
    with open(l1_loss_file, "a") as f:
        f.write('edge loss:   ' + str(edge_loss/len(image_re_paths)) + "\n" + "\n")


def print_StyleAndContent(class_path, args):
    l1_loss_file = os.path.join(args.res_dir, "loss.txt")
    RE_Path = class_path
    GT_Path = os.path.join(args.data_dir, args.test_dataset, args.target_path)
    Content_Path = os.path.join(args.data_dir, args.test_dataset, args.target_path)
    GT_list = sorted([os.path.join(GT_Path, i) for i in os.listdir(GT_Path)])[: 972]
    RE_list = sorted([os.path.join(RE_Path, i) for i in os.listdir(RE_Path) if not i.__contains__('mean')])[: 972]
    Content_list = sorted([os.path.join(Content_Path, i) for i in os.listdir(Content_Path)])[: 972]
    sty_sum, cont_sum = 0.0, 0.0
    for i in range(len(RE_list)):
        print(i)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = prepare_model(device)
        optimizing_img = prepare_img(RE_list[i], 128, device)
        current_set_of_feature_maps = neural_net(optimizing_img)
        ########################################################################################
        content_img = prepare_img(Content_list[i], 128, device)
        content_img_set_of_feature_maps = neural_net(content_img)
        gt_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
        current_content_representation = current_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
        content_loss = torch.nn.MSELoss(reduction='mean')(gt_content_representation, current_content_representation)
        ########################################################################################
        style_gt_img = prepare_img(GT_list[i], 128, device)
        style_gt_img_set_of_feature_maps = neural_net(style_gt_img)
        gt_style_representation = [gram_matrix(x) for cnt, x in enumerate(style_gt_img_set_of_feature_maps) if
                                   cnt in style_feature_maps_indices_names[0]]
        current_style_representation = [gram_matrix(x) for cnt, x in enumerate(current_set_of_feature_maps) if
                                        cnt in style_feature_maps_indices_names[0]]
        style_loss = 0.0
        for gram_gt, gram_hat in zip(gt_style_representation, current_style_representation):
            style_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
        style_loss /= len(gt_style_representation)
        ########################################################################################
        sty_sum += style_loss
        cont_sum += content_loss
    with open(l1_loss_file, "a") as f:
        f.write('Style Loss:   ' + str(sty_sum / len(RE_list)) + "\n")
        f.write('Content Loss:   ' + str(cont_sum / len(RE_list)) + "\n")


def print_FID(class_path, args):
    pathA = class_path
    pathGT = os.path.join(args.data_dir, args.test_dataset, args.target_path)
    l1_loss_file = os.path.join(args.res_dir, "loss.txt")
    a_dataset = FIDDataset(pathA, 'mean')
    o_dataset = FIDDataset(pathA, 'one')
    d_dataset = FIDDataset(pathGT)
    FID_mean = compute_FID(a_dataset, d_dataset)
    FID_one = compute_FID(o_dataset, d_dataset)
    with open(l1_loss_file, "a") as f:
        f.write('FID mean:'+
            str(FID_mean) + "\t" + "\t" + 'FID one:' + str(FID_one) + "\n")


def validateUN(data_loader, networks, epoch, args, additional=None):
    # set nets
    G = networks['G'].cuda(args.gpu)
    C_glyph = networks['C_glyph'].cuda(args.gpu)
    C_effect = networks['C_effect'].cuda(args.gpu)
    G_EMA = networks['G_EMA'].cuda(args.gpu)
    # switch to train mode
    G.eval()
    C_glyph.eval()
    C_effect.eval()
    G_EMA.eval()
    # data loader
    val_dataset = data_loader['VALSET']

    # Parse images for average reference vector
    x_each_cls = []

    refs_times = 1
    with torch.no_grad():
        val_tot_tars = torch.tensor(val_dataset.targets_final)
        len_g = val_dataset.len_g
        len_t = val_dataset.len_t
        num_tmp_val = len(val_tot_tars) // (len_g * len_t)
        args.att_to_use = [i for i in range(len_g * len_t)]
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)
        print("init x_each_cls")
        for cls_idx in range(len(args.att_to_use)):
            print("init x_each_cls"+str(cls_idx))
            tmp_cls_set = (val_tot_tars == args.att_to_use[cls_idx]).nonzero()
            c = torch.randint(0, 1, (refs_times * args.val_batch, 1))
            for i in range(refs_times * args.val_batch):
                b = random.randint(0, num_tmp_val-1)
                c[i] = tmp_cls_set[b]

            #####################################
            # Cheating = [573, 615, 122, 581]
            # for ch in range(len(Cheating)):
            #     c[ch][0] = tmp_cls_set[Cheating[ch]-1]
            #####################################
            tmp_ds = torch.utils.data.Subset(val_dataset, c)
            tmp_dl = torch.utils.data.DataLoader(tmp_ds, batch_size=args.val_batch, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)  # shuffle=False
            tmp_iter = iter(tmp_dl)
            tmp_sample = None
            for imgs, _ in tmp_iter:
                x_ = imgs
                if tmp_sample is None:
                    tmp_sample = x_.clone()
                else:
                    tmp_sample = torch.cat((tmp_sample, x_), 0)
            x_each_cls.append(tmp_sample)
        # for i in range(len(x_each_cls)):
        #     vutils.save_image(x_each_cls[i], os.path.join(args.res_dir,
        #                                             '{}_EMA_{}.jpg'.format(args.gpu, i)), normalize=True, nrow=6)
        print("test......")
        epoch_path = os.path.join(args.res_dir, str(epoch) + "epoch")
        if not os.path.exists(epoch_path):
            os.mkdir(epoch_path)

        for i in range(len(x_each_cls)):
            for j in range(len(x_each_cls)):
                if i % len_t == 0 and j < len_t:
                    class_path = os.path.join(epoch_path, str(i // len_g)+'_'+str(j % len_t))
                    if not os.path.exists(class_path):
                        os.mkdir(class_path)
                    for content_index, (_, source) in enumerate(val_loader):
                        source = source.cuda(args.gpu, non_blocking=True)
                        if source.size(0) < args.val_batch:
                            source_new = torch.randn([args.val_batch, 3, 128, 128]).cuda(args.gpu, non_blocking=True)
                            for k in range(source.size(0)):
                                source_new[k] = source[k]
                            source = source_new
                        glyph_refs_class = i
                        effect_refs_class = j
                        glyph_input = x_each_cls[glyph_refs_class].cuda(args.gpu, non_blocking=True)
                        effect_input = x_each_cls[effect_refs_class].cuda(args.gpu, non_blocking=True)
                        vec_source = G_EMA.cnt_encoder(source)
                        #########################################
                        vec_glyph = C_glyph(glyph_input, sty=True)
                        vec_effect = C_effect(effect_input, sty=True)
                        # vec_glyph = C_glyph.attention(glyph_input, glyph_input.unsqueeze(1).repeat(1, 3, 1, 1, 1), 'G')
                        # vec_effect = C_effect.attention(effect_input, effect_input.unsqueeze(1).repeat(1, 3, 1, 1, 1), 'G')
                        #########################################
                        x_res_ema_tmp = G_EMA.decode(vec_glyph, vec_effect, vec_source)  # B X 512 X 16 X 16

                        mean_glyph_code = torch.mean(vec_glyph, dim=0)
                        mean_glyph_code = mean_glyph_code.repeat((vec_glyph.size(0), 1))
                        mean_effect_code = torch.mean(vec_effect, dim=0)
                        mean_effect_code = mean_effect_code.repeat((vec_effect.size(0), 1))
                        x_res_ema_tmp_mean = G_EMA.decode(mean_glyph_code, mean_effect_code, vec_source)

                        if not class_path.__contains__('2_2'):
                            refs_output = torch.cat((source, glyph_input), 0)
                            refs_output = torch.cat((refs_output, effect_input), 0)
                            refs_output_one = torch.cat((refs_output, x_res_ema_tmp), 0)
                            refs_output_mean = torch.cat((refs_output, x_res_ema_tmp_mean), 0)
                            vutils.save_image(refs_output_one, os.path.join(class_path, '{}_{}_{}_{}_{}.jpg'.format(args.gpu, epoch, content_index, i, j)), normalize=True, nrow=args.val_batch)
                            vutils.save_image(refs_output_mean, os.path.join(class_path, '{}_{}_{}_{}_{}_mean.jpg'.format(args.gpu, epoch,content_index,i, j)),normalize=True, nrow=args.val_batch)
                            break
                        for k in range(x_res_ema_tmp.size(0)):
                            if content_index * args.batch_size + k + 1 > 52:
                                break
                            vutils.save_image(x_res_ema_tmp[k], os.path.join(class_path, '{}.jpg'.format(str(content_index * args.batch_size + k + 1).zfill(4))), normalize=True)
                            vutils.save_image(x_res_ema_tmp_mean[k], os.path.join(class_path, '{}_mean.jpg'.format(str(content_index * args.batch_size + k + 1).zfill(4))), normalize=True)

                        if content_index * args.batch_size + k + 1 > 52:
                            break
                    if class_path.__contains__('2_2'):
                        print('print_L1......')
                        print_L1(class_path, args, epoch)
                        print('print_FID......')
                        print_FID(class_path, args)
                        print('print_SSIM_PNSR......')
                        print_SSIM_PNSR(class_path, args)
                        print('print_StyleAndContent......')
                        print_StyleAndContent(class_path, args)
                        print('print_EdgeLoss......')
                        print_EdgeLoss(class_path, args)

