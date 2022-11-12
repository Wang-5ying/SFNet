import os
import torch
import torch.nn.functional as F
import pytorch_iou
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from dataset import get_loader, test_dataset
from utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from config import opt
from torch.cuda import amp
# set the device for training
cudnn.benchmark = True
cudnn.enabled = True


os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU:', opt.gpu_id)

# build the model
#from rgbt.rgbt_models.LENet import LENetmobilenetv2evo
from DSLRDNet.model import build_model
cfg="train"

model = build_model('resnet')
print('model:Trans_Teacher')
if (opt.load is not None):
    model.load_state_dict(torch.load(opt.load))
    print('load model from ', opt.load)

model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

# set the path
train_dataset_path = opt.train_root
image_root = train_dataset_path  + '/RGB/'
ti_root = train_dataset_path  + '/T/'
gt_root = train_dataset_path  + '/GT/'

# val_dataset_path = opt.val_root
# val_image_root = val_dataset_path  + '/RGB/'
# val_ti_root = val_dataset_path  + '/T/'
# val_gt_root = val_dataset_path  + '/GT/'

# 保存训练权重
save_path = opt.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
print('load data...')
train_loader = get_loader(image_root, gt_root, ti_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
# test_loader = test_dataset(val_image_root, val_gt_root, val_ti_root, opt.trainsize)
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
IOU = pytorch_iou.IOU(size_average = True).cuda()

step = 0
writer = SummaryWriter(save_path + 'summary', flush_secs = 30)
best_mae = 1
best_epoch = 0
Sacler = amp.GradScaler()




# train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step,best_mae,best_epoch
    model.train()
    loss_all = 0
    epoch_step = 0
    mae_sum = 0
    # best_mae = 1
    # best_epoch = 0
    try:
        for i, (images, gts, ti) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = images.cuda()
            ti = ti.cuda()
            gts = gts.cuda()
            # gts2 = F.interpolate(gts, (112, 112))
            # gts3 = F.interpolate(gts, (56, 56))
            # gts4 = F.interpolate(gts, (28, 28))
            # gts5 = F.interpolate(gts, (14, 14))

            # bound = tesnor_bound(gts, 3).cuda()
            # bound2 = F.interpolate(bound, (112, 112))
            # bound3 = F.interpolate(bound, (56, 56))
            # bound4 = F.interpolate(bound, (28, 28))
            # bound5 = F.interpolate(bound, (14, 14))
            # print(depths.shape)
            #out = model(images, ti)
            s1,s2,s3 = model(images)
            loss=0
            for i in s1:
                loss+=CE(i,gts)
            for i in s2:
                loss+=CE(i,gts)
            for i in s3:
                loss+=CE(i,gts)
            #loss1 = CE(out, gts) + IOU()

            # loss7 = CE(s1[6], gts)

            # loss11 = CE(s1[10], gts)
            # loss2 = CE(s1[1], gts)


            loss = loss1+loss2+loss3+loss4
            # loss = loss1

            res = torch.sigmoid(s1[0])
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_train = torch.sum(torch.abs(res - gts)) * 1.0 / (torch.numel(gts))
            mae_sum = mae_train.item() + mae_sum
            # supvision
            # loss2 = CE(out[1], gts3)
            # loss3 = CE(out[2], gts4)
            # predict_bound0 = out[0]
            # predict_bound1 = out[1]
            # predict_bound2 = out[2]
            # # predict_bound3 = out[3]
            # predict_bound0 = tesnor_bound(torch.sigmoid(predict_bound0), 3)
            # predict_bound1 = tesnor_bound(torch.sigmoid(predict_bound1), 3)
            # predict_bound2 = tesnor_bound(torch.sigmoid(predict_bound2), 3)
            # predict_bound3 = tesnor_bound(torch.sigmoid(predict_bound3), 3)
            # loss6 = IOUBCEWithoutLogits(predict_bound0, bound)
            # loss7 = IOUBCEWithoutLogits(predict_bound1, bound3)
            # loss8 = IOUBCEWithoutLogits(predict_bound2, bound4)
            # print(loss1,loss2,loss3,loss4,loss5,loss6,loss7,loss8)
            #loss = loss1 #+ loss2 + loss3
            # loss = loss1 + loss8
            # print(loss.data[0])
            loss.backward()
            # Sacler.scale(loss).backward()
            # clip_gradient(optimizer, opt.clip)
            # Sacler.step(optimizer)
            # Sacler.update()
            optimizer.step()
            step = step + 1
            epoch_step = epoch_step + 1
            loss_all = loss.item() + loss_all
            if i % 10 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.item()))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                             format(epoch, opt.epoch, i, total_step, loss.item()))
                writer.add_scalar('Loss', loss, global_step=step)
                # grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                # writer.add_image('train/RGB', grid_image, step)
                # grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                # writer.add_image('train/Ground_truth', grid_image, step)
                # grid_image = make_grid(ti[0].clone().cpu().data, 1, normalize=True)
                # writer.add_image('train/depths', grid_image, step)
                # grid_image = make_grid(bound[0].clone().cpu().data, 1, normalize=True)
                # writer.add_image('train/bounds', grid_image, step)
                # res = out[0][0].clone()
                # res = res.sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('OUT/final', torch.tensor(res), step, dataformats='HW')
                # res = predict_bound0[0].clone()
                # res = res.sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('OUT/saptical', torch.tensor(res), step, dataformats='HW')
                # res = out[3][0].clone()
                # res = res.data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('OUT/struct', torch.tensor(res), step, dataformats='HW')
                # res = predict_bound2[0].clone()
                # res = res.sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('OUT/saptical3', torch.tensor(res), step, dataformats='HW')

        loss_all /= epoch_step
        mae = mae_sum / len(train_loader)
        writer.add_scalar('MAE', mae, global_step=epoch)
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:

                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'best_epoch.pth')

        print('Epoch: {} MAE: {} bestmae: {} bestepoch: {}'.format(epoch, mae, best_mae, best_epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))
        # logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise


# # test function
# def test(test_loader, model, epoch, save_path):
#     global best_mae, best_epoch
#     model.eval()
#     with torch.no_grad():
#         mae_sum = 0
#         for i in range(test_loader.size):
#             image, gt, ti, name = test_loader.load_data()
#             gt = gt.cuda()
#             image = image.cuda()
#             ti = ti.cuda()
#             with amp.autocast():
#                 res = model(image, ti)
#                 res = torch.sigmoid(res[0])
#                 res = (res - res.min()) / (res.max() - res.min() + 1e-8)
#                 mae_train = torch.sum(torch.abs(res - gt)) * 1.0 / (torch.numel(gt))
#                 # print(mae_train)
#                 mae_sum = mae_train.item() + mae_sum
#         # print(test_loader.size)
#         mae = mae_sum / test_loader.size
#         # print(test_loader.size)
#         writer.add_scalar('MAE', torch.as_tensor(mae), global_step=epoch)
#         print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
#         if epoch == 1:
#             best_mae = mae
#         else:
#             if mae < best_mae:
#                 best_mae = mae
#                 best_epoch = epoch
#                 torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
#                 print('best epoch:{}'.format(epoch))
#         logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        # print([image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')])
        # print(f for f in os.listdir(image_root) if f.endswith('.jpg'))
        train(train_loader, model, optimizer, epoch, save_path)
        # test(test_loader, model, epoch, save_path)
