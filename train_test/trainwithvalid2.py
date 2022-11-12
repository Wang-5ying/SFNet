#muqian zuihaode
import argparse
import os
import torch
import torchvision.models
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

import pytorch_iou
from datetime import datetime

from dataset import get_loader, test_dataset
# from newsedge22.test import test_loader
from utils import adjust_lr
# from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from config import opt
from torch.cuda import amp
# from SwinMCNetmain.loss.ssim import SSIM
# set the device for training
cudnn.benchmark = True
cudnn.enabled = True

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU:', opt.gpu_id)
# build the model
print("hhh")
# from AMajorchanges.xiugai3withoutedge.demo7trcedgegaijin22233normyuanchangedirection2212225backboneunetnewpdm import SRAA
from second_model.IENet.bayibest82segformerbest1011.newresdecoder4a614t4615622xiuz74711715726 import M

cfg = "train"

print(torch.__version__)
model = M()
# model = SwinMCNet()
print('model:Trans_Teacher')
if (opt.load is not None):
    model.load_state_dict(torch.load(opt.load))
    print('load model from ', opt.load)

model.cuda()


model.load_pre("/home/wby/segformer.b5.640x640.ade.160k.pth") #!!!!
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

# set the path
train_dataset_path = opt.train_root
image_root = train_dataset_path + '/RGB/'
ti_root = train_dataset_path + '/T/'
gt_root = train_dataset_path + '/GT/'
bound_root = train_dataset_path + '/bound/'

# monodepth_root = train_dataset_path + '/output_monodepth/'
body_root = train_dataset_path + '/bodyskeleton/'
detail_root = train_dataset_path + '/detailcontour/'
val_dataset_path = opt.val_root
val_image_root = val_dataset_path  + '/RGB/'
val_ti_root = val_dataset_path  + '/T/'
val_gt_root = val_dataset_path  + '/GT/'

# 保存训练权重
save_path = opt.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
print('load data...')
train_loader = get_loader(image_root, gt_root, ti_root, bound_root,body_root,detail_root,
                          batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(val_image_root, val_gt_root, val_ti_root, opt.trainsize)
total_step = len(train_loader)
print(total_step)

logging.basicConfig(filename=save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("Model: LHDecoder4_small_catorignal2")
logging.info(save_path + "Train")
logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
        opt.decay_epoch))

# set loss function
CE = torch.nn.BCEWithLogitsLoss().cuda()
IOU = pytorch_iou.IOU(size_average=True).cuda()
KL = nn.KLDivLoss(reduction='mean')
def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()
# from SwinMCNetmain.loss.ssim import SSIM
# ssim_loss = SSIM(window_size=11, size_average=True)
step = 0
# writer = SummaryWriter(save_path + 'summary', flush_secs = 30)
best_mae = 1
best_epoch = 0
Sacler = amp.GradScaler()

# ##############################################
# train function
def train(train_loader, model, optimizer, epoch, save_path, temperature):
    global step, best_mae, best_epoch
    model.train()
    loss_all = 0
    epoch_step = 0
    mae_sum = 0

    # best_mae = 1
    # best_epoch = 0
    try:
        for i, (images, gts, ti, bounds,body,details) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = images.cuda()
            ti = ti.cuda()
            gts = gts.cuda()
            bounds = bounds.cuda()

            body = body.cuda()
            detail = details.cuda()


            s = model(images, ti, 1)
            # DepthNet's output

            # loss4 = CE(s[8], gts)
            loss1 = CE(s[0], gts)
            loss2 = CE(s[1], gts)
            loss3 = CE(s[2], gts)
            loss4 = CE(s[3], gts)

            loss = loss1 + loss2 + loss3 + loss4


            res = torch.sigmoid(s[0])
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_train = torch.sum(torch.abs(res.cuda() - gts.cuda())) * 1.0 / (torch.numel(gts.cuda()))
            mae_sum = mae_train.item() + mae_sum

            torch.cuda.empty_cache()
            loss.backward()

            optimizer.step()
            step = step + 1
            epoch_step = epoch_step + 1
            loss_all = loss.item() + loss_all
            if i % 10 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.item()))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                             format(epoch, opt.epoch, i, total_step, loss.item()))

        loss_all /= epoch_step
        mae = mae_sum / len(train_loader)
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'best_epoch.pth')

        print('Epoch: {} MAE: {} bestmae: {} bestepoch: {}'.format(epoch, mae, best_mae, best_epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))
        if epoch % 5 == 0:
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise


# test function
def val(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, ti, name = test_loader.load_data()
            gt = gt.cuda()
            image = image.cuda()
            ti = ti.cuda()
            with amp.autocast():
                res = model(image, ti)
                res = torch.sigmoid(res[0])
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                mae_train = torch.sum(torch.abs(res - gt)) * 1.0 / (torch.numel(gt))
                # print(mae_train)
                mae_sum = mae_train.item() + mae_sum
        # print(test_loader.size)
        mae = mae_sum / test_loader.size
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start train...")
    global temperature
    # temperature = 34          #!!!!!!!!!!!!!!!!!!!
    temperature = 1
    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, save_path, temperature)
        print('temperature:', temperature)
        print("lr",cur_lr)

