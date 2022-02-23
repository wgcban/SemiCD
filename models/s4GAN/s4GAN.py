import math, time
from itertools import chain
import torch
import torch.nn.functional as F
from torch import nn
from base import BaseModel
from utils.helpers import set_trainable
from utils.losses import *
from models.decoders import *
from models.encoder import Encoder
from utils.losses import CE_loss
from torch.autograd import Variable
from models.s4GAN.discriminators import s4GAN_discriminator, find_good_maps, loss_calc, one_hot

class s4GAN(BaseModel):
    def __init__(self, num_classes, conf, sup_loss=None, cons_w_unsup=None, testing=False,
            pretrained=True, use_weak_lables=False, weakly_loss_w=0.4):

        self.num_classes = num_classes
        if not testing:
            assert (sup_loss is not None) and (cons_w_unsup is not None)

        super(s4GAN, self).__init__()
        assert int(conf['supervised']) + int(conf['semi']) == 1, 'one mode only'
        if conf['supervised']:
            self.mode = 'supervised'
        else:
            self.mode = 'semi'

        # Supervised and unsupervised losses
        self.unsuper_loss   = nn.CrossEntropyLoss()
        
        self.unsup_loss_w   = cons_w_unsup
        self.sup_loss_w     = conf['supervised_w']
        self.softmax_temp   = conf['softmax_temp']
        self.sup_loss       = sup_loss
        self.sup_type       = conf['sup_loss']

        # Use weak labels
        self.use_weak_lables= use_weak_lables
        self.weakly_loss_w  = weakly_loss_w
        # pair wise loss (sup mat)
        self.aux_constraint     = conf['aux_constraint']
        self.aux_constraint_w   = conf['aux_constraint_w']
        # confidence masking (sup mat)
        self.confidence_th      = conf['confidence_th']
        self.confidence_masking = conf['confidence_masking']

        # Create the model
        self.encoder = Encoder(pretrained=pretrained)

        # The main encoder
        upscale             = 8
        num_out_ch          = 2048
        decoder_in_ch       = num_out_ch // 4
        self.main_decoder   = MainDecoder(upscale, decoder_in_ch, num_classes=num_classes)

        # The auxilary decoders
        if self.mode == 'semi' or self.mode == 'weakly_semi':
            self.criterionD = nn.BCELoss()
            self.model_D = s4GAN_discriminator(num_classes=2)


    def forward(self, A_l=None, B_l=None, target_l=None, A_ul=None, B_ul=None, target_ul=None, curr_iter=None, epoch=None):
        if not self.training:
            return self.main_decoder(self.encoder(A_l, B_l))

        # We compute the losses in the forward pass to avoid problems encountered in muti-gpu 

        # Forward pass the labels example
        input_size  = (A_l.size(2), A_l.size(3))
        output_l    = self.main_decoder(self.encoder(A_l, B_l))
        if output_l.shape != A_l.shape:
            output_l = F.interpolate(output_l, size=input_size, mode='bilinear', align_corners=True)

        # Supervised loss
        if self.sup_type == 'CE':
            loss_sup = self.sup_loss(output_l, target_l, temperature=self.softmax_temp) * self.sup_loss_w 
        elif self.sup_type == 'FL':
            loss_sup = self.sup_loss(output_l,target_l) * self.sup_loss_w
        else:
            loss_sup = self.sup_loss(output_l, target_l, curr_iter=curr_iter, epoch=epoch) * self.sup_loss_w

        # If supervised mode only, return
        if self.mode == 'supervised':
            curr_losses = {'loss_sup': loss_sup}
            outputs = {'sup_pred': output_l}
            total_loss = loss_sup
            return total_loss, curr_losses, outputs

        # If semi supervised mode
        elif self.mode == 'semi':
            # Get main prediction
            x_ul      = self.encoder(A_ul, B_ul)
            output_ul = self.main_decoder(x_ul)

            ### Compute unsupervised loss s4GAN ###
            ### From s4GAN
            #For unlabeled data
            for param in self.model_D.parameters():
                param.requires_grad = False
            A_ul_d = (A_ul-torch.min(A_ul))/(torch.max(A_ul)- torch.min(A_ul))
            B_ul_d = (B_ul-torch.min(B_ul))/(torch.max(B_ul)- torch.min(B_ul))
            pred_cat = torch.cat((F.softmax(output_ul, dim=1), A_ul_d, B_ul_d), dim=1)
            D_out_z, D_out_y_pred = self.model_D(pred_cat)
            pred_sel, labels_sel, count = find_good_maps(D_out_z, output_ul)
            if count > 0:
                loss_st = loss_calc(pred_sel, labels_sel)
            else:
                loss_st = 0.0
            #For labeled data
            D_gt_v = Variable(one_hot(target_l)).cuda()
            A_l = (A_l - torch.min(A_l))/(torch.max(A_l)-torch.min(A_l))
            B_l = (B_l - torch.min(B_l))/(torch.max(B_l)-torch.min(B_l))
            D_gt_v_cat = torch.cat((D_gt_v, A_l, B_l), dim=1)
            D_out_z_gt , D_out_y_gt = self.model_D(D_gt_v_cat)
            # L1 loss for Feature Matching Loss
            loss_fm = torch.mean(torch.abs(torch.mean(D_out_y_gt, 0) - torch.mean(D_out_y_pred, 0)))
            if count > 0: # if any good predictions found for self-training loss
                loss_sup = loss_sup +  0.1*loss_fm + 1.0*loss_st 
            else:
                loss_sup = loss_sup + 0.1*loss_fm
            curr_losses = {'loss_sup': loss_sup}

            #Training Descriminator
            # train D
            for param in self.model_D.parameters():
                param.requires_grad = True
            # train with pred
            pred_cat = pred_cat.detach()
            D_out_z, _ = self.model_D(pred_cat)
            y_fake_ = Variable(torch.zeros(D_out_z.size(0), 1).cuda())
            loss_D_fake = self.criterionD(D_out_z, y_fake_)
            # train with gt
            D_out_z_gt , _ = self.model_D(D_gt_v_cat)
            y_real_ = Variable(torch.ones(D_out_z_gt.size(0), 1).cuda()) 
            loss_D_real = self.criterionD(D_out_z_gt, y_real_)
        
            loss_unsup = (loss_D_fake + loss_D_real)/2.0

            if output_ul.shape != A_ul.shape:
                output_ul = F.interpolate(output_ul, size=input_size, mode='bilinear', align_corners=True)
            outputs = {'sup_pred': output_l, 'unsup_pred': output_ul}

            # Compute the unsupervised loss
            weight_u    = self.unsup_loss_w(epoch=epoch, curr_iter=curr_iter)
            loss_unsup  = loss_unsup * weight_u
            curr_losses['loss_unsup'] = loss_unsup
            total_loss  = loss_unsup  + loss_sup 

            return total_loss, curr_losses, outputs

    def get_backbone_params(self):
        return self.encoder.get_backbone_params()

    def get_other_params(self):
        if self.mode == 'semi':
            return chain(self.encoder.get_module_params(), self.main_decoder.parameters(), 
                        self.model_D.parameters())

        return chain(self.encoder.get_module_params(), self.main_decoder.parameters())

