import math, time
from itertools import chain
import torch
import torch.nn.functional as F
from torch import nn
from base import BaseModel
from utils.helpers import set_trainable
from utils.losses import *
from models.decoder import *
from models.encoder import Encoder, DiffModule
from models.rotation import RotationPredHead, RotationPredHeadSim

from utils.losses import CE_loss

class ResNet50_RoCD(BaseModel):
    def __init__(self, num_classes, conf, sup_loss=None, cons_w_unsup=None, testing=False,
            pretrained=True, N_temp_rots=None):

        self.num_classes = num_classes
        if not testing:
            assert (sup_loss is not None) and (cons_w_unsup is not None)

        super(ResNet50_RoCD, self).__init__()
        assert int(conf['supervised']) + int(conf['semi']) == 1, 'one mode only'
        if conf['supervised']:
            self.mode = 'supervised'
        else:
            self.mode = 'semi'

        # Supervised and unsupervised losses
        self.RotationLoss  = torch.nn.CrossEntropyLoss()
        
        self.unsup_loss_w   = cons_w_unsup
        self.sup_loss_w     = conf['supervised_w']
        
        self.sup_loss       = sup_loss
        self.sup_type       = conf['sup_loss']

        # Create the model
        self.encoder    = Encoder(pretrained=pretrained)
        self.DiffModule = DiffModule()

        # The main encoder
        upscale             = 8
        num_out_ch          = 2048
        decoder_in_ch       = num_out_ch // 4
        self.main_decoder   = ConvDecoder(upscale, decoder_in_ch, num_classes=num_classes)

        # Initializing the rotation prediction task
        if self.mode == 'semi':
            print('>>> Self-supervised temporal rotation prediction for semi-supervised CD <<<')
            self.N_temp_rots   = N_temp_rots
            self.rot_pred_head = RotationPredHeadSim(emb_dim=num_out_ch, N_temp_rots=self.N_temp_rots)

    def forward(self, A_l=None, B_l=None, target_l=None, A_l_r=None, B_l_r=None, target_l_r=None, A_ul=None, B_ul=None, target_ul=None, A_ul_r=None, B_ul_r=None, target_ul_r=None, curr_iter=None, epoch=None):
        if not self.training:
            return self.main_decoder(self.DiffModule(self.encoder(A_l), self.encoder(B_l)))
        # We compute the losses in the forward pass to avoid problems encountered in muti-gpu 

        # Forward pass the labels example
        input_size  = (A_l.size(2), A_l.size(3)) 
        output_l    = self.main_decoder(self.DiffModule(self.encoder(A_l), self.encoder(B_l)))
        if output_l.shape != A_l.shape:
            output_l = F.interpolate(output_l, size=input_size, mode='bilinear', align_corners=True)
        #Compute the ratget
        cm_l = output_l.detach().argmax(1).to(torch.float32)

        # Supervised loss
        if self.sup_type == 'CE':
            loss_sup = self.sup_loss(output_l, target_l, temperature=1.0) * self.sup_loss_w 
        elif self.sup_type == 'FL':
            loss_sup = self.sup_loss(output_l,target_l) * self.sup_loss_w
        else:
            loss_sup = self.sup_loss(output_l, target_l, curr_iter=curr_iter, epoch=epoch) * self.sup_loss_w

        # If supervised mode only, return
        if self.mode    == 'supervised':
            curr_losses = {'loss_sup':loss_sup}
            outputs     = {'sup_pred': output_l}
            total_loss  = loss_sup
            return total_loss, curr_losses, outputs

        # If semi supervised mode: utilizing rotation prediction as an auxilary task
        elif self.mode == 'semi':
            # Get prediction for unlabeled data
            output_ul   = self.main_decoder(self.DiffModule(self.encoder(A_ul), self.encoder(B_ul)))
            # Generate targets
            cm_ul   = output_ul.detach().argmax(1).to(torch.float32)

            # Rotation prediction for semi-supevised learning
            r_l         = self.rot_pred_head(self.encoder(A_l_r), self.encoder(B_l_r), cm_l, target_l_r)
            r_ul        = self.rot_pred_head(self.encoder(A_ul_r), self.encoder(B_ul_r), cm_ul, target_l_r)
            loss_unsup  = self.RotationLoss(r_l, target_l_r) + self.RotationLoss(r_ul, target_ul_r)
            
            # Supervised loss
            curr_losses = {'loss_sup': loss_sup}

            # Predicted output for labeled and unlabeled data
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
        return self.encoder.parameters()

    def get_other_params(self):
        if self.mode == 'semi':
            return chain(self.DiffModule.parameters(), self.main_decoder.parameters(), 
                        self.rot_pred_head.parameters())
        else:
            return chain(self.DiffModule.parameters(), self.main_decoder.parameters())

