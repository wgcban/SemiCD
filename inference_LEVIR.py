import argparse
import scipy, math
from scipy import ndimage
import cv2
import numpy as np
import sys
import json
import models
import dataloaders
from utils.helpers import colorize_mask
from utils.pallete import get_voc_pallete
from utils import metrics
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
from math import ceil
from PIL import Image
from pathlib import Path
from utils.metrics import eval_metrics, AverageMeter
from utils.htmlwriter import HTML
from matplotlib import pyplot as plt
from utils.helpers import DeNormalize
import time


class testDataset(Dataset):
    def __init__(self, Dataset_Path):
        mean    = [0.485, 0.456, 0.406]
        std     = [0.229, 0.224, 0.225]
        self.Dataset_Path = Dataset_Path

        file_list       = os.path.join(self.Dataset_Path, 'list', "train.txt")
        self.filelist   = np.loadtxt(file_list, dtype=str)
        if self.filelist.ndim == 2:
            return self.filelist[:, 0]

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        image_A_path    = os.path.join(self.Dataset_Path, 'A', self.filelist[index])
        image_B_path    = os.path.join(self.Dataset_Path, 'B', self.filelist[index])
        label_path      = os.path.join(self.Dataset_Path, 'label', self.filelist[index])

        image_A         = np.asarray(Image.open(image_A_path), dtype=np.float32)
        image_B         = np.asarray(Image.open(image_B_path), dtype=np.float32)
        label           = np.asarray(Image.open(label_path), dtype=np.int32)
        image_id        = self.filelist[index].split("/")[-1].split(".")[0]

        image_A = self.normalize(self.to_tensor(image_A))
        image_B = self.normalize(self.to_tensor(image_B))
        return image_A, image_B, label, image_id

def multi_scale_predict(model, image_A, image_B, scales, num_classes, flip=False):
    H, W        = (image_A.size(2), image_A.size(3))
    upsize      = (ceil(H / 8) * 8, ceil(W / 8) * 8)
    upsample    = nn.Upsample(size=upsize, mode='bilinear', align_corners=True)
    pad_h, pad_w= upsize[0] - H, upsize[1] - W
    image_A     = F.pad(image_A, pad=(0, pad_w, 0, pad_h), mode='reflect')
    image_B     = F.pad(image_B, pad=(0, pad_w, 0, pad_h), mode='reflect')

    total_predictions = np.zeros((num_classes, image_A.shape[2], image_A.shape[3]))

    for scale in scales:
        scaled_img_A = F.interpolate(image_A, scale_factor=scale, mode='bilinear', align_corners=False)
        scaled_img_B = F.interpolate(image_B, scale_factor=scale, mode='bilinear', align_corners=False)
        scaled_prediction = upsample(model(A_l=scaled_img_A, B_l=scaled_img_B))
        
        if flip:
            fliped_img_A = scaled_img_A.flip(-1)
            fliped_img_B = scaled_img_B.flip(-1)
            fliped_predictions  = upsample(model(A_l=fliped_img_A, B_l=fliped_img_B))
            scaled_prediction   = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)
        total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)

    total_predictions /= len(scales)
    return total_predictions[:, :H, :W]

def main():
    args = parse_arguments()

    # CONFIG
    assert args.config
    config = json.load(open(args.config))
    scales = [1.0]

    # DATA
    testdataset = testDataset(args.Dataset_Path)
    loader      = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=1)
    num_classes = 2
    palette     = get_voc_pallete(num_classes)

    # MODEL
    config['model']['supervised'] = True
    config['model']['semi'] = False
    model = models.Consistency_ResNet50_CD(num_classes=num_classes, conf=config['model'], testing=True)
    print(f'\n{model}\n')
    checkpoint = torch.load(args.model)
    model = torch.nn.DataParallel(model)
    try:
        print("Loading the state dictionery...")
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    except Exception as e:
        print(f'Some modules are missing: {e}')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    model.cuda()

    if args.save and not os.path.exists('outputs'):
        os.makedirs('outputs')
    
    #Set HTML
    web_dir = '/media/lidan/ssd2/SemiCD/outputs/'+config["experim_name"]
    html_results = HTML(web_dir=web_dir, exp_name=config['experim_name']+"--Test--",
                            save_name=config['experim_name'], config=config)

    # LOOP OVER THE DATA
    tbar = tqdm(loader, ncols=100)
    total_inter, total_union = 0, 0
    total_correct, total_label = 0, 0

    for index, data in enumerate(tbar):
        image_A, image_B, label, image_id = data
        image_A = image_A.cuda()
        image_B = image_B.cuda()
        label   = label.cuda()
        
        #PREDICT
        with torch.no_grad():
            output = multi_scale_predict(model, image_A, image_B, scales, num_classes)
        prediction = np.asarray(np.argmax(output, axis=0), dtype=np.uint8)

        #Calculate metrics
        output = torch.from_numpy(output).cuda()
        label[label>=1] = 1
        output = torch.unsqueeze(output, 0)
        label  = torch.unsqueeze(label, 0)
        correct, labeled, inter, union  = eval_metrics(output, label, num_classes)
        total_inter, total_union        = total_inter+inter, total_union+union
        total_correct, total_label      = total_correct+correct, total_label+labeled
        pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
        IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
        tbar.set_description('Test Results | PixelAcc: {:.2f}, IoU(no-change): {:.2f}, IoU(change): {:.2f} |'.format(pixAcc, IoU[0], IoU[1]))

        #SAVE RESULTS
        prediction_im = colorize_mask(prediction, palette)
        prediction_im.save('/media/lidan/ssd2/SemiCD/outputs/'+config["experim_name"]+'/'+image_id[0]+'.png')
    
    #Printing average metrics on test-data
    pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
    IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
    mIoU = IoU.mean()
    seg_metrics = {"Pixel_Accuracy": np.round(pixAcc, 3), "Mean_IoU": np.round(mIoU, 3),
                                "Class_IoU": dict(zip(range(num_classes), np.round(IoU, 3)))}
    log = {     
                'val_loss': 0.0,
                **seg_metrics
            }
    html_results.add_results(epoch=1, seg_resuts=log)
    html_results.save()

def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--config', default='/media/lidan/ssd2/SemiCD/saved/SemiCD_v1/config.json',type=str,
                        help='Path to the config file')
    parser.add_argument( '--model', default='/media/lidan/ssd2/SemiCD/saved/SemiCD_v1/best_model.pth', type=str,
                        help='Path to the trained .pth model')
    parser.add_argument( '--save', action='store_true', help='Save images')
    parser.add_argument('--Dataset_Path', default="/media/lidan/ssd2/CDData/LEVIR-CD256", type=str,
                        help='Path to dataset LEVIR-CD')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()

