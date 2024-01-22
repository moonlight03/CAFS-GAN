"""
TUNIT: Truly Unsupervised Image-to-Image Translation
Copyright (c) 2020-present NAVER Corp.
MIT license 
"""
from torch.nn import functional as F
from tqdm import trange
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tools.utils import *
from tools.ops import compute_grad_gp, update_average, copy_norm_params, calc_adv_loss, calc_recon_loss


def trainGAN_SUP(data_loader, networks, opts, epoch, args, additional):
    # avg meter
    d_losses = AverageMeter()
    d_advs = AverageMeter()
    d_gps = AverageMeter()

    g_losses = AverageMeter()
    g_advs = AverageMeter()
    g_styconts = AverageMeter()

    moco_losses = AverageMeter()

    # set nets
    D_glyph = networks['D_glyph'].cuda(args.gpu)
    D_effect = networks['D_effect'].cuda(args.gpu)
    G = networks['G'].cuda(args.gpu)
    C_glyph = networks['C_glyph'].cuda(args.gpu)
    C_effect = networks['C_effect'].cuda(args.gpu)
    G_EMA = networks['G_EMA'].cuda(args.gpu)

    # set opts
    d_glyph_opt = opts['D_glyph']
    d_effect_opt = opts['D_effect']
    g_opt = opts['G']
    c_glyph_opt = opts['C_glyph']
    c_effect_opt = opts['C_effect']
    # switch to train mode
    D_glyph.train()
    D_effect.train()
    G.train()
    C_glyph.train()
    C_effect.train()
    G_EMA.train()

    # logger = additional['logger']


    # summary writer
    train_it = iter(data_loader)

    t_train = trange(0, args.iters, initial=0, total=args.iters)

    for i in t_train:
        refs1, refs2, target1, target2, source, X_aug, Y_aug, X_negs, Y_negs = next(train_it)
        source = source.cuda(args.gpu)
        glyph_refs = refs1.cuda(args.gpu)
        effect_refs = refs2.cuda(args.gpu)
        target1 = target1.cuda(args.gpu)
        target2 = target2.cuda(args.gpu)
        X_aug = X_aug.cuda(args.gpu)
        Y_aug = Y_aug.cuda(args.gpu)
        X_negs = X_negs.cuda(args.gpu)
        Y_negs = Y_negs.cuda(args.gpu)
        ####################
        # BEGIN Train Encoders #
        ####################
        zeros = torch.zeros(glyph_refs.size(0), dtype=torch.long).cuda(args.gpu)
        vec_glyph = C_glyph.moco(glyph_refs)
        neg = None
        for j in range(X_negs.size(1)):
            if neg == None:
                neg = torch.bmm(vec_glyph.view(vec_glyph.size(0), 1, -1), C_glyph.moco(X_negs[:, j, :, :, :]).view(vec_glyph.size(0), -1, 1)).view(vec_glyph.size(0), 1)
            else:
                neg = torch.cat([neg, torch.bmm(vec_glyph.view(vec_glyph.size(0), 1, -1), C_glyph.moco(X_negs[:, j, :, :, :]).view(vec_glyph.size(0), -1, 1)).view(vec_glyph.size(0), 1)], 1)
        pos = torch.bmm(vec_glyph.view(vec_glyph.size(0), 1, -1), C_glyph.moco(X_aug).view(vec_glyph.size(0), -1, 1)).view(vec_glyph.size(0), 1)
        logit = torch.cat([pos, neg], 1)
        loss = F.cross_entropy(logit / 0.07, zeros)
        glyph_loss = loss
        c_glyph_opt.zero_grad()
        glyph_loss.backward()
        c_glyph_opt.step()

        zeros1 = torch.zeros(glyph_refs.size(0), dtype=torch.long).cuda(args.gpu)
        vec_effect = C_effect.moco(effect_refs)
        neg = None
        for j in range(Y_negs.size(1)):
            if neg == None:
                neg = torch.bmm(vec_effect.view(vec_effect.size(0), 1, -1), C_effect.moco(Y_negs[:, j, :, :, :]).view(vec_effect.size(0), -1, 1)).view(vec_effect.size(0), 1)
            else:
                neg = torch.cat([neg, torch.bmm(vec_effect.view(vec_effect.size(0), 1, -1),C_effect.moco(Y_negs[:, j, :, :, :]).view(vec_effect.size(0), -1, 1)).view(vec_effect.size(0), 1)], 1)
        pos = torch.bmm(vec_effect.view(vec_effect.size(0), 1, -1), C_effect.moco(Y_aug).view(vec_effect.size(0), -1, 1)).view(vec_effect.size(0), 1)
        logit = torch.cat([pos, neg], 1)
        loss = F.cross_entropy(logit / 0.07, zeros1)
        effect_loss = loss
        c_effect_opt.zero_grad()
        effect_loss.backward()
        c_effect_opt.step()

        ####################
        # BEGIN Train D #
        ####################
        with torch.no_grad():
            vec_glyph = C_glyph.moco_attention(glyph_refs, X_aug)
            vec_effect = C_effect.moco_attention(effect_refs, Y_aug)
            vec_source = G.cnt_encoder(source)
            x_fake = G.decode(vec_glyph, vec_effect, vec_source)
        ##########################################################
        glyph_refs.requires_grad_()
        d_real_logit_glyph, _ = D_glyph(glyph_refs, target1, 0)
        d_fake_logit_glyph, _ = D_glyph(x_fake.detach(), target1, 0)
        d_adv_real_glyph = calc_adv_loss(d_real_logit_glyph, 'd_real')
        d_adv_fake_glyph = calc_adv_loss(d_fake_logit_glyph, 'd_fake')
        d_gp_glyph = args.w_gp * compute_grad_gp(d_real_logit_glyph, glyph_refs, is_patch=False)

        d_glyph_opt.zero_grad()
        d_adv_real_glyph.backward(retain_graph=True)
        d_gp_glyph.backward()
        d_adv_fake_glyph.backward()
        d_glyph_opt.step()
        ##########################################################
        effect_refs.requires_grad_()
        d_real_logit_effect, _ = D_effect(effect_refs, target2, 1)
        d_fake_logit_effect, _ = D_effect(x_fake.detach(), target2, 1)
        d_adv_real_effect = calc_adv_loss(d_real_logit_effect, 'd_real')
        d_adv_fake_effect = calc_adv_loss(d_fake_logit_effect, 'd_fake')
        d_gp_effect = args.w_gp * compute_grad_gp(d_real_logit_effect, effect_refs, is_patch=False)

        d_effect_opt.zero_grad()
        d_adv_real_effect.backward(retain_graph=True)
        d_gp_effect.backward()
        d_adv_fake_effect.backward()
        d_effect_opt.step()
        #########################################################
        ####################
        # BEGIN Train G #
        ####################
        vec_glyph = C_glyph.moco_attention(glyph_refs, X_aug)
        vec_effect = C_effect.moco_attention(effect_refs,Y_aug)

        vec_source = G.cnt_encoder(source)
        x_fake = G.decode(vec_glyph, vec_effect, vec_source)

        g_fake_logit_glyph, _ = D_glyph(x_fake, target1, 0)
        g_fake_logit_effect, _ = D_effect(x_fake, target2, 1)

        g_adv_fake_glyph = calc_adv_loss(g_fake_logit_glyph, 'g')
        g_adv_fake_effect = calc_adv_loss(g_fake_logit_effect, 'g')

        g_adv = g_adv_fake_glyph + g_adv_fake_effect

        g_loss = args.w_adv * g_adv

        g_opt.zero_grad()
        c_glyph_opt.zero_grad()
        c_effect_opt.zero_grad()
        g_loss.backward()

        c_glyph_opt.step()
        c_effect_opt.step()
        g_opt.step()

        ##################
        # END Train GANs #
        ##################

        if epoch >= args.ema_start:
            update_average(G_EMA, G)

        torch.cuda.synchronize()

        with torch.no_grad():
            if (i + 1) % args.log_step == 0 and (args.gpu == 0 or args.gpu == '0'):
                # summary_step = epoch * args.iters + i
                # add_logs(args, logger, 'D/LOSS', d_losses.avg, summary_step)
                # add_logs(args, logger, 'D/ADV', d_advs.avg, summary_step)
                # add_logs(args, logger, 'D/GP', d_gps.avg, summary_step)
                #
                # add_logs(args, logger, 'G/LOSS', g_losses.avg, summary_step)
                # add_logs(args, logger, 'G/ADV', g_advs.avg, summary_step)
                # add_logs(args, logger, 'G/STYCONT', g_styconts.avg, summary_step)
                #
                # add_logs(args, logger, 'C/MOCO', moco_losses.avg, summary_step)

                print('Epoch: [{}/{}] [{}/{}] Avg Loss: D[{d_losses.avg:.2f}] G[{g_losses.avg:.2f}] '
                      'C[{moco_losses.avg:.2f}]'.format(epoch + 1, args.epochs, i+1, args.iters
                                                        , d_losses=d_losses, g_losses=g_losses,
                                                        moco_losses=moco_losses))

    copy_norm_params(G_EMA, G)

