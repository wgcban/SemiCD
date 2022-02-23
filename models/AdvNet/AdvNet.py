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
from models.AdvNet.discriminators import FCDiscriminator,make_D_label
from utils.losses import BCEWithLogitsLoss2d

def loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda()
    criterion = CrossEntropy2d().cuda()

    return criterion(pred, label)

class AdvNet(BaseModel):
    def __init__(self, num_classes, conf, sup_loss=None, cons_w_unsup=None, testing=False,
            pretrained=True, use_weak_lables=False, weakly_loss_w=0.4):

        self.num_classes = num_classes
        if not testing:
            assert (sup_loss is not None) and (cons_w_unsup is not None)

        super(AdvNet, self).__init__()
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
            bce_loss = BCEWithLogitsLoss2d()
            self.model_D = FCDiscriminator(num_classes=2, ndf = 64)


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

            # Compute adversarial loss
            pred_remain = output_ul.detach()
            D_out       = self.model_D(F.softmax(output_ul))
            D_out_sigmoid = F.sigmoid(D_out).data.cpu().numpy().squeeze(axis=1)
            ignore_mask_remain = np.zeros(D_out_sigmoid.shape).astype(np.bool)
            loss_adv = bce_loss(D_out, make_D_label(1, ignore_mask_remain)).data.cpu().numpy()[0]

            #Computing unsupervised loss
            semi_ignore_mask = (D_out_sigmoid < 0.3)
            semi_gt = output_ul.data.cpu().numpy().argmax(axis=1)
            semi_gt[semi_ignore_mask] = 255

            semi_ratio = 1.0 - float(semi_ignore_mask.sum())/semi_ignore_mask.size
            if semi_ratio == 0.0:
                loss_semi_value += 0
            else:
                semi_gt = torch.FloatTensor(semi_gt)
                
                loss_semi = loss_calc(output_ul, semi_gt)
                loss_semi = loss_semi
                loss_semi_value += loss_semi.data.cpu().numpy()[0]

            loss_unsup = loss_adv+ loss_semi_value
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
                        self.model_D.parameters())

        return chain(self.encoder.get_module_params(), self.main_decoder.parameters())

