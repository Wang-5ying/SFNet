import time

import torch
import sys
from ptflops import get_model_complexity_info
sys.path.append('./models')
import os
import cv2

from second_model.IENet.bayibest82segformerbest1011.newresdecoder4a614t4615622xiuz74711715726 import M

cfg = "train"

model = M()
from config import opt
from dataset import test_dataset

dataset_path = opt.test_path

# set device for test
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU:', opt.gpu_id)

# load the model
# model = UTA(cfg="train")
# Large epoch size may not generalize well. You can choose a good model to load according to the log file and pth files saved in ('./BBSNet_cpts/') when training.
model.load_state_dict(torch.load('second_model/IENet/Net_epoch_best.pth'))

model.cuda()
model.eval()

# test
test_mae = []
test_datasets = ['VT800', 'VT5000', 'VT1000']



class averagemeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def paras(model):
    total = sum(p.numel for p in model. parameters())
    print('params: %.2fM' % (total/1e6))


for dataset in test_datasets:
    mae_sum = 0
    save_path = '/home/wby/PycharmProjects/new sedge2/output/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    ti_root = dataset_path + dataset + '/T/'
    test_loader = test_dataset(image_root, gt_root, ti_root, opt.testsize)
    time_meter = averagemeter()
    for i in range(test_loader.size):
        time_start = time.time()
        with torch.no_grad():
            image, gt, ti, name = test_loader.load_data()
            gt = gt.cuda()
            # print(gt.type())
            image = image.cuda()
            ti = ti.cuda()
            res = model(image, ti, 1)
            predict = torch.sigmoid(res[0])
            predict = (predict - predict.min()) / (predict.max() - predict.min() + 1e-8)
            mae = torch.sum(torch.abs(predict - gt)) / torch.numel(gt)
            mae_sum = mae.item() + mae_sum
            time_meter.update(time.time() - time_start, n=test_loader.size)
        predict = predict.data.cpu().numpy().squeeze()
        # print(predict.shape)
        print('save img to: ', save_path + name)
        cv2.imwrite(save_path + name, predict * 255)
    test_mae.append(mae_sum / test_loader.size)
print('Test Done!', 'MAE', test_mae)
print('inference fps: ', 1/time_meter.avg)
b, c, h, w = image.shape
flops, params = get_model_complexity_info(model, (b, c, h, w), as_strings=True, print_per_layer_stat=False)
print('Flops', flops)
print('Params: ', params)
