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
from models.SemiCDNet_TGRS21.discriminators import FCDiscriminator

class SemiCDNet_TGRS21(BaseModel):
    def __init__(self, num_classes, conf, sup_loss=None, cons_w_unsup=None, testing=False,
            pretrained=True, use_weak_lables=False, weakly_loss_w=0.4):

        self.num_classes = num_classes
        if not testing:
            assert (sup_loss is not None) and (cons_w_unsup is not None)

        super(SemiCDNet_TGRS21, self).__init__()
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
            self.Ds = FCDiscriminator(num_classes=1)
            self.De = FCDiscriminator(num_classes=2)


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

            # Compute unsupervised loss
            loss_Ds = self.unsuper_loss(self.Ds(torch.unsqueeze(target_l,1)), torch.ones(output_l.size(0),1,1).type(torch.LongTensor).cuda()) + self.unsuper_loss(self.Ds(torch.argmax(output_ul.detach(), dim=1, keepdim=True)), torch.zeros(output_ul.size(0),1,1).type(torch.LongTensor).cuda())
            E_l     =  F.softmax(output_l.detach(), dim=1)*torch.log(F.softmax(output_l.detach(), dim=1)+1e-6)
            E_ul    =  F.softmax(output_ul.detach(), dim=1)*torch.log(F.softmax(output_ul.detach(), dim=1)+1e-6)
            loss_De   = self.unsuper_loss(self.De(E_l), torch.ones(output_l.size(0),1,1).type(torch.LongTensor).cuda()) + self.unsuper_loss(self.De(E_ul), torch.zeros(output_ul.size(0),1,1).type(torch.LongTensor).cuda())
            loss_unsup = loss_Ds  + loss_De
            curr_losses = {'loss_sup': loss_sup}

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
                        self.Ds.parameters(), self.De.parameters())

        return chain(self.encoder.get_module_params(), self.main_decoder.parameters())

