"""
TUNIT: Truly Unsupervised Image-to-Image Translation
Copyright (c) 2020-present NAVER Corp.
MIT license 
"""
import argparse
import warnings
from datetime import datetime
from glob import glob
from shutil import copyfile
from collections import OrderedDict

import torch.nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from models.generator_stylegan2 import Generator as Generator
from models.discriminator import Discriminator as Discriminator
from models.guidingNet import GuidingNet

from train.train_unsupervised import trainGAN_UNSUP
from train.train_semisupervised import trainGAN_SEMI
from train.train_supervised_aug import trainGAN_SUP

from validation.validation_mean import validateUN

from tools.utils import *
from datasets.datasetgetter import get_dataset


# Configuration
parser = argparse.ArgumentParser(description='PyTorch GAN Training')
parser.add_argument('--train_dataset', default='compare_8')
parser.add_argument('--test_dataset', default='compare_9')
parser.add_argument('--content_path', default='SP_SSKS-30')
parser.add_argument('--target_path', default='ZCP_XLL-30')
parser.add_argument('--train_imagenums', default=774, type=int)
parser.add_argument('--test_contentimagenums', default=972, type=int)
parser.add_argument('--workers', default=8, type=int, help='the number of workers of data loader')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size for training')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use.')

parser.add_argument('--dataset', default='animal_faces', help='Dataset name to use',
                    choices=['afhq_cat', 'afhq_dog', 'afhq_wild', 'animal_faces', 'photo2ukiyoe', 'summer2winter', 'lsun_car', 'ffhq'])
parser.add_argument('--data_path', type=str, default='../data_font71',
                    help='Dataset directory. Please refer Dataset in README.md')
parser.add_argument('--model_name', type=str, default='GAN',
                    help='Prefix of logs and results folders. '
                         'ex) --model_name=ABC generates ABC_20191230-131145 in logs and results')
parser.add_argument('--epochs', default=100, type=int, help='Total number of epochs to run. Not actual epoch.')
parser.add_argument('--iters', default=1000, type=int, help='Total number of iterations per epoch')
parser.add_argument('--val_batch', default=8, type=int,
                    help='Batch size for validation. '
                         'The result images are stored in the form of (val_batch, val_batch) grid.')
parser.add_argument('--log_step', default=100, type=int)
parser.add_argument('--sty_dim', default=128, type=int, help='The size of style vector')
parser.add_argument('--output_k', default=3, type=int, help='Total number of classes to use')
parser.add_argument('--img_size', default=128, type=int, help='Input image size')
parser.add_argument('--dims', default=2048, type=int, help='Inception dims for FID')
parser.add_argument('--p_semi', default=1.0, type=float,
                    help='Ratio of labeled data '
                         '0.0 = unsupervised mode'
                         '1.0 = supervised mode'
                         '(0.0, 1.0) = semi-supervised mode')
parser.add_argument('--load_model', default=None, type=str, metavar='PATH',
                    help='GAN_20220922-215819'
                         'ex) --load_model GAN_20190101_101010'
                         'It loads the latest .ckpt file specified in checkpoint.txt in GAN_20190101_101010')
parser.add_argument('--validation', dest='validation', action='store_true',
                    help='Call for valiation only mode')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--ddp', dest='ddp', action='store_true', help='Call if using DDP')
parser.add_argument('--port', default='8989', type=str)

parser.add_argument('--iid_mode', default='iid+', type=str, choices=['iid', 'iid+'])

parser.add_argument('--w_gp', default=10.0, type=float, help='Coefficient of GP of D')
parser.add_argument('--w_rec', default=0.1, type=float, help='Coefficient of Rec. loss of G')
parser.add_argument('--w_adv', default=1.0, type=float, help='Coefficient of Adv. loss of G')
parser.add_argument('--w_vec', default=0.01, type=float, help='Coefficient of Style vector rec. loss of G')


def main():
    ####################
    # Default settings #
    ####################
    args = parser.parse_args()
    print("PYTORCH VERSION", torch.__version__)
    args.data_dir = args.data_path
    args.start_epoch = 0

    assert (args.p_semi >= 0.0) and (args.p_semi <= 1.0)

    # p_semi = 0.0 : unsupervised
    # p_semi = 1.0 : supervised
    # p_semi = 0.0~1.0 : semi-
    if args.p_semi == 0.0:
        args.train_mode = 'GAN_UNSUP'
    elif args.p_semi == 1.0:
        args.train_mode = 'GAN_SUP'
    else:
        args.train_mode = 'GAN_SEMI'

    den = args.iters//1000

    # unsup_start : train networks with supervised data only before unsup_start
    # separated : train IIC only until epoch = args.separated
    # ema_start : Apply EMA to Generator after args.ema_start
    if args.train_mode in ['GAN_SEMI']:
        args.unsup_start = 20
        args.separated = 35
        args.ema_start = 36
        args.fid_start = 36
    elif args.train_mode in ['GAN_UNSUP']:
        args.unsup_start = 0
        args.separated = 65
        args.ema_start = 66
        args.fid_start = 66
    elif args.train_mode in ['GAN_SUP']:
        args.unsup_start = 0
        args.separated = 0
        args.ema_start = 1
        args.fid_start = 0

    args.unsup_start = args.unsup_start // den  # 同上，都没变
    args.separated = args.separated // den
    args.ema_start = args.ema_start // den
    args.fid_start = args.fid_start // den

    # Cuda Set-up
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    args.multiprocessing_distributed = False

    args.distributed = False

    ngpus_per_node = torch.cuda.device_count()
    args.ngpus_per_node = ngpus_per_node

    print("MULTIPROCESSING DISTRIBUTED : ", args.multiprocessing_distributed)

    # Logs / Results
    if args.load_model is None:
        args.model_name = '{}_{}'.format(args.model_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        args.model_name = args.load_model

    makedirs('./logs')
    makedirs('./results')

    args.log_dir = os.path.join('./logs', args.model_name)
    args.event_dir = os.path.join(args.log_dir, 'events')
    args.res_dir = os.path.join('./results', args.model_name)

    makedirs(args.log_dir)
    dirs_to_make = next(os.walk('./'))[1]
    not_dirs = ['.idea', '.git', 'logs', 'results', '.gitignore', '.nsmlignore', 'resrc']
    makedirs(os.path.join(args.log_dir, 'codes'))
    for to_make in dirs_to_make:
        if to_make in not_dirs:
            continue
        makedirs(os.path.join(args.log_dir, 'codes', to_make))
    makedirs(args.res_dir)

    if args.load_model is None:
        pyfiles = glob("./*.py")
        for py in pyfiles:
            copyfile(py, os.path.join(args.log_dir, 'codes') + "/" + py)

        for to_make in dirs_to_make:
            if to_make in not_dirs:
                continue
            tmp_files = glob(os.path.join('./', to_make, "*.py"))
            for py in tmp_files:
                copyfile(py, os.path.join(args.log_dir, 'codes', py[2:]))


    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = 0
    # # of GT-classes
    args.num_cls = args.output_k

    # Classes to use

    args.att_to_use = [0, 1, 2, 3, 4, 5, 6, 7, 8]


    # IIC statistics
    args.epoch_acc = []
    args.epoch_avg_subhead_acc = []
    args.epoch_stats = []

    # Logging

    # build model - return dict
    networks, opts = build_model(args)

    # load model if args.load_model is specified
    load_model(args, networks, opts)
    cudnn.benchmark = True

    # get dataset and data loader
    train_dataset, val_dataset = get_dataset(args.dataset, args)  # 训练集是10类，验证集是不同的10类50张
    train_loader, val_loader, train_sampler = get_loader(args, {'train': train_dataset, 'val': val_dataset})

    # map the functions to execute - un / sup / semi-
    trainFunc, validationFunc = map_exec_func(args)  # 有三个文件，挑选文件

    # print all the argument
    print_args(args)

    # For saving the model
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        record_txt = open(os.path.join(args.log_dir, "record.txt"), "a+")
        for arg in vars(args):
            record_txt.write('{:35}{:20}\n'.format(arg, str(getattr(args, arg))))
        record_txt.close()

    # Run
    validationFunc(val_loader, networks, 0, args)
    # save_model(args, 0, networks, opts)

    for epoch in range(args.start_epoch, args.epochs):
        print("START EPOCH[{}]".format(epoch+1))
        #if (epoch + 1) >= 40 and (epoch + 1) % 10 == 0:
        #    save_model(args, epoch, networks, opts)
        if epoch == args.ema_start and 'GAN' in args.train_mode:
            networks['G_EMA'].load_state_dict(networks['G'].state_dict())
        trainFunc(train_loader, networks, opts, epoch, args, {'logger': None})
        if (epoch + 1) % 10 == 0:
            validationFunc(val_loader, networks, epoch + 1, args)


#################
# Sub functions #
#################
def print_args(args):
    for arg in vars(args):
        print('{:35}{:20}\n'.format(arg, str(getattr(args, arg))))


def build_model(args):
    args.to_train = 'CDGI'

    networks = {}
    opts = {}
    is_semi = (0.0 < args.p_semi < 1.0)
    if is_semi:
        assert 'SEMI' in args.train_mode
    if 'C' in args.to_train:
        networks['C_glyph'] = GuidingNet(args.img_size, {'cont': args.sty_dim, 'disc': args.output_k})
        networks['C_effect'] = GuidingNet(args.img_size, {'cont': args.sty_dim, 'disc': args.output_k})
    if 'D' in args.to_train:
        networks['D_glyph'] = Discriminator(args.img_size, num_domains=args.output_k)
        networks['D_effect'] = Discriminator(args.img_size, num_domains=args.output_k)
    if 'G' in args.to_train:
        networks['G'] = Generator(args.img_size, args.sty_dim, use_sn=False)
        networks['G_EMA'] = Generator(args.img_size, args.sty_dim, use_sn=False)
    if 'I' in args.to_train:
        networks['inceptionNet'] = None
    print("Use GPU: {} for training".format(args.gpu))

    if 'C' in args.to_train:
        opts['C_glyph'] = torch.optim.Adam(
            networks['C_glyph'].module.parameters() if args.distributed else networks['C_glyph'].parameters(),
            1e-4, weight_decay=0.001)
        opts['C_effect'] = torch.optim.Adam(
            networks['C_effect'].module.parameters() if args.distributed else networks['C_effect'].parameters(),
            1e-4, weight_decay=0.001)
    if 'D' in args.to_train:
        opts['D_glyph'] = torch.optim.RMSprop(
            networks['D_glyph'].module.parameters() if args.distributed else networks['D_glyph'].parameters(),
            1e-4, weight_decay=0.0001)
        opts['D_effect'] = torch.optim.RMSprop(
            networks['D_effect'].module.parameters() if args.distributed else networks['D_effect'].parameters(),
            1e-4, weight_decay=0.0001)
    if 'G' in args.to_train:
        opts['G'] = torch.optim.RMSprop(
            networks['G'].module.parameters() if args.distributed else networks['G'].parameters(),
            1e-4, weight_decay=0.0001)

    return networks, opts


def load_model(args, networks, opts):
    if args.load_model is not None:
        check_load = open(os.path.join(args.log_dir, "checkpoint.txt"), 'r')
        to_restore = check_load.readlines()[-1].strip()
        load_file = os.path.join(args.log_dir, to_restore)
        if os.path.isfile(load_file):
            print("=> loading checkpoint '{}'".format(load_file))
            checkpoint = torch.load(load_file, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            if not args.multiprocessing_distributed:
                for name, net in networks.items():
                    if name in ['inceptionNet']:
                        continue
                    tmp_keys = next(iter(checkpoint[name + '_state_dict'].keys()))
                    if 'module' in tmp_keys:
                        tmp_new_dict = OrderedDict()
                        for key, val in checkpoint[name + '_state_dict'].items():
                            tmp_new_dict[key[7:]] = val
                        net.load_state_dict(tmp_new_dict)
                        networks[name] = net
                    else:
                        net.load_state_dict(checkpoint[name + '_state_dict'])
                        networks[name] = net

            for name, opt in opts.items():
                opt.load_state_dict(checkpoint[name.lower() + '_optimizer'])
                opts[name] = opt
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(load_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.log_dir))


def get_loader(args, dataset):
    train_dataset = dataset['train']
    val_dataset = dataset['val']

    print(len(val_dataset))

    # GAN_IIC_SEMI

    train_dataset_ = train_dataset
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset_, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    val_loader = {'VAL': val_loader, 'VALSET': val_dataset}
    return train_loader, val_loader, train_sampler


def map_exec_func(args):
    if args.train_mode == 'GAN_SUP':
        trainFunc = trainGAN_SUP
        validationFunc = validateUN
    elif args.train_mode == 'GAN_UNSUP':
        trainFunc = trainGAN_UNSUP
        validationFunc = validateUN
    elif args.train_mode == 'GAN_SEMI':
        trainFunc = trainGAN_SEMI
        validationFunc = validateUN
    else:
        exit(-6)

    return trainFunc, validationFunc


def save_model(args, epoch, networks, opts):
    check_list = open(os.path.join(args.log_dir, "checkpoint.txt"), "a+")
    with torch.no_grad():
        save_dict = {}
        save_dict['epoch'] = epoch + 1
        for name, net in networks.items():
            if name in ['inceptionNet']:
                continue
            save_dict[name + '_state_dict'] = net.state_dict()
            if name != 'G_EMA':
                save_dict[name.lower()+'_optimizer'] = opts[name].state_dict()
        print("SAVE CHECKPOINT[{}] DONE".format(epoch+1))
        save_checkpoint(save_dict, check_list, args.log_dir, epoch + 1)
    check_list.close()


if __name__ == '__main__':
    main()
