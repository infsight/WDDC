import torch
import argparse
import os

import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
from PIL import Image
import random
from torchvision import transforms
import time
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import cv2

from wddc.wddc_model import Difseg


class BBCEWithLogitLoss(nn.Module):
    '''
    Balanced BCEWithLogitLoss
    '''
    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)

        return loss


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        num_pos = torch.sum(mask)
        num_neg = mask.numel() - num_pos
        w_pos = (1 - p) ** self.gamma
        w_neg = p ** self.gamma

        loss_pos = -self.alpha * mask * w_pos * torch.log(p + 1e-12)
        loss_neg = -(1 - self.alpha) * (1 - mask) * w_neg * torch.log(1 - p + 1e-12)

        loss = (torch.sum(loss_pos) + torch.sum(loss_neg)) / (num_pos + num_neg + 1e-12)

        return loss


def _iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    # pred[pred >= 0.5] = 1.
    # pred[pred < 0.5] = 0.
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - ((inter + 1e-15) / (union + 1e-15))

    return iou.mean()


def cv_random_flip(img, label):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label


def calc_mean_std(img_tensor):
    mean = img_tensor.view(3, -1).mean(dim=1)
    std = img_tensor.view(3, -1).std(dim=1)
    return mean, std


def adainaug(content_img, style_img):
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    aug_flag = random.randint(0, 1)
    img = content_img
    if aug_flag == 1:
        content = to_tensor(content_img)
        style = to_tensor(style_img)
        c_mean, c_std = calc_mean_std(content)
        s_mean, s_std = calc_mean_std(style)
        normalized = (content - c_mean.view(3, 1, 1)) / (c_std.view(3, 1, 1) + 1e-5)
        stylized = normalized * s_std.view(3, 1, 1) + s_mean.view(3, 1, 1)
        stylized = stylized.clamp(0, 1)
        img = to_pil(stylized)
    return img


class GetData(Dataset):

    # 初始化为整个class提供全局变量，为后续方法提供一些量
    def __init__(self, img_dir, mask_dir, transformsize=None, train=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.train = train
        self.img_path_list = sorted(os.listdir(self.img_dir))
        self.mask_path_list = sorted(os.listdir(self.mask_dir))
        
        # self.style_img_dir = 'autodl-tmp/diffusion/aug_image'
        self.style_img_dir = 'autodl-tmp/diffusion/inp2p'
        self.style_img_list = sorted(os.listdir(self.style_img_dir))
        
        self.img_transform = transforms.Compose([
            transforms.Resize((transformsize, transformsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
                transforms.Resize((transformsize, transformsize), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()])
            
    def __getitem__(self, idx):
        assert len(self.img_path_list) == len(self.mask_path_list), f'{len(self.img_path_list)} != {len(self.mask_path_list)}'
        img_name = self.img_path_list[idx]  # 只获取了文件名
        mask_name = self.mask_path_list[idx]
        img_item_path = os.path.join(self.img_dir, img_name) # 每个图片的位置
        mask_item_path = os.path.join(self.mask_dir, mask_name)
        img = Image.open(img_item_path).convert("RGB")
        gt = Image.open(mask_item_path).convert("L")
        if self.train is not None:
            img, gt = cv_random_flip(img, gt)
            style_name = random.choice(self.style_img_list)
            style_path = os.path.join(self.style_img_dir, style_name) # 每个图片的位置
            style_img = Image.open(style_path).convert("RGB")
            img = adainaug(img, style_img)

        img = self.img_transform(img)
        gt = self.gt_transform(gt)
        gt[gt > 0] = 1.

        return img, gt,  mask_name

    def __len__(self):
        return len(self.img_path_list)


def eval_epoch(model, val_loader, results_file, device):
    model.eval()
    
    hardcase_dir = 'autodl-tmp/UW-Bench-v2-4/test/Hardcase'
    hardcase_list = sorted(os.listdir(hardcase_dir))
    
    mask_dir = 'autodl-tmp/UW-Bench-v2-4/test/SegmentationClass'
    pbar = tqdm(total=len(val_loader), leave=True, desc='val')
    pixel_TP, pixel_TP_hardcase = 0, 0
    pixel_TP_and_FP, pixel_TP_and_FP_hardcase = 0, 0
    pixel_TP_and_FN, pixel_TP_and_FN_hardcase = 0, 0
    with torch.no_grad():
        for inp, gt, mask_name in val_loader:
            inp = inp.to(device)
            pred = model(inp)
            
            ori_gt = cv2.imread(os.path.join(mask_dir, mask_name[0]), 0)
            # ori_gt = np.array(Image.open(os.path.join(mask_dir, mask_name[0])).convert('L'))
            ori_gt[ori_gt > 0] = 1.
            current_pixel_TP_and_FN = ori_gt.sum()

            pred = torch.sigmoid(pred)
            pred = transforms.functional.resize(pred,  ori_gt.shape[:2], transforms.InterpolationMode.BILINEAR)
            pred = pred.squeeze().detach().cpu().numpy()
            pred = np.where(pred > 0.5, 1, 0)

            current_pixel_TP_and_FP = pred.sum()

            current_pixel_TP = (ori_gt * pred).sum()

            pixel_TP += current_pixel_TP
            pixel_TP_and_FP += current_pixel_TP_and_FP
            pixel_TP_and_FN += current_pixel_TP_and_FN
            
            if mask_name[0] in hardcase_list:
                pixel_TP_hardcase += current_pixel_TP
                pixel_TP_and_FP_hardcase += current_pixel_TP_and_FP
                pixel_TP_and_FN_hardcase += current_pixel_TP_and_FN

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()
        
        if pixel_TP_and_FP == 0:
            pixel_precision = 0
        else:
            pixel_precision = pixel_TP / pixel_TP_and_FP
        pixel_recall = pixel_TP / pixel_TP_and_FN
        mIoU = pixel_TP / (pixel_TP_and_FP + pixel_TP_and_FN - pixel_TP)

        if pixel_TP_and_FP_hardcase == 0:
            pixel_precision_hardcase = 0
        else:
            pixel_precision_hardcase = pixel_TP_hardcase / pixel_TP_and_FP_hardcase
        pixel_recall_hardcase = pixel_TP_hardcase / pixel_TP_and_FN_hardcase
        mIoU_hardcase = pixel_TP_hardcase / (pixel_TP_and_FP_hardcase + pixel_TP_and_FN_hardcase - pixel_TP_hardcase)

        val_info = f'precison: {pixel_precision:.4f}, recall: {pixel_recall:.4f}, miou: {mIoU:.4f}' + '\n'
        val_info = val_info + f'precision_hardcase: {pixel_precision_hardcase:.4f}, recall_hardcase: {pixel_recall_hardcase:.4f}, miou_hardcase: {mIoU_hardcase:.4f}'
        print(val_info)
        results_file.write(val_info + '\n')
        results_file.flush()
    return mIoU, mIoU_hardcase


def train_epoch(train_loader, model, optimizer, device):
    model.train()

    pbar = tqdm(total=len(train_loader), leave=True, desc='train')
    bce_loss = torch.nn.BCEWithLogitsLoss()
    focal_loss = FocalLoss()

    losses_G, losses_bce, losses_iou, losses_focal = 0, 0, 0, 0
    i = 0
    for inp, gt, mask_name in train_loader:
        inp = inp.to(device)
        gt = gt.to(device)

        pred = model(inp)

        loss_bce = bce_loss(pred, gt)
        loss_iou = _iou_loss(pred, gt)
        loss_focal = focal_loss(pred, gt)

        loss_G = loss_bce + loss_iou + loss_focal

        optimizer.zero_grad()  # set G's gradients to zero
        loss_G.backward()  # calculate graidents for G
        optimizer.step()  # udpate G's weights

        losses_G += loss_G.item()
        losses_bce += loss_bce.item()
        losses_iou += loss_iou.item()
        losses_focal += loss_focal.item()
        i += 1

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    mloss_G = losses_G / i
    mloss_bce = losses_bce / i
    mloss_iou = losses_iou / i
    mloss_focal = losses_focal / i
    return mloss_G, mloss_bce, mloss_iou, mloss_focal


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--savepath', default='autodl-tmp/diffusion/save', help='')
args = parser.parse_args()

global log_info
device = torch.device(args.device)
print(f'Using device {device}')

train_data = GetData('autodl-tmp/UW-Bench-v2-4/train_supervised/JPEGImages',
                     'autodl-tmp/UW-Bench-v2-4/train_supervised/SegmentationClass',
                     transformsize=512, train=True)
print('train data size:', len(train_data))
train_loader = DataLoader(dataset=train_data, batch_size=2, shuffle=True, num_workers=8)
val_data = GetData('autodl-tmp/UW-Bench-v2-4/test/JPEGImages',
                   'autodl-tmp/UW-Bench-v2-4/test/SegmentationClass', transformsize=512)
print('test data size:', len(val_data))
val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=8)

context_num = 6
model = Difseg(context_num=context_num).to(device)
print('trainable params:')
for name, para in model.named_parameters():
    if para.requires_grad:
        print(name)
model_total_params = sum(p.numel() for p in model.parameters())
model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

optimizer = optim.AdamW(model.parameters(), lr=0.0001)
max_epoch = 20
lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=1.0e-7)

threshold = 0.5
epoch_start = 1
epoch_max = 20
epoch_val = 1
best_iou = 0
best_iou_hard = 0
log_testing = open(os.path.join(args.savepath, 'log_wddc_6context_5timestep_resize_ori.txt'), 'w')
for epoch in range(epoch_start, epoch_max + 1):
    log_info = []
    train_loss_G, train_loss_bce, train_loss_iou, train_loss_focal = train_epoch(train_loader, model, optimizer, device)
    lr_scheduler.step()

    log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
    log_info.append('train: loss_G: loss={:.4f}, loss_bce={:.4f}, loss_iou={:.4f}, loss_focal={:.4f}'.format(train_loss_G, train_loss_bce, train_loss_iou, train_loss_focal))
    print(log_info)
    log_testing.write(str(log_info) + '\n')
    log_testing.flush()

    if (epoch_val is not None) and (epoch % epoch_val == 0):
        mIoU, mIoU_hardcase = eval_epoch(model, val_loader, log_testing, device)
        if mIoU > best_iou:
            best_iou = mIoU
            torch.save(model.state_dict(), os.path.join(args.savepath, 'wddc_6context_5timestep_resize_ori_best.pth'))
        # if mIoU_hardcase > best_iou_hard:
        #     best_iou_hard = mIoU_hardcase
        #     torch.save(model.state_dict(), os.path.join(args.savepath, 'wddc_6context_hard_best.pth'))
print('done!')
